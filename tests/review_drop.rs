#![cfg(all(feature = "std", feature = "compiler"))]

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use miniscript::policy::Concrete;
use miniscript::{AbsLockTime, Miniscript, MiniscriptKey, Tap, Terminal, Threshold};

#[test]
fn compile_and_drop_deep_policy() {
    let mut prefixes = vec![Arc::new(Concrete::Key("A".to_owned()))];
    for _ in 0..256 {
        let child = Arc::clone(prefixes.last().unwrap());
        prefixes.push(Arc::new(Concrete::And(vec![
            child,
            Arc::new(Concrete::After(AbsLockTime::from_consensus(1).unwrap())),
        ])));
    }
    let policy = Arc::clone(prefixes.last().unwrap());
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(move || {
            let ms = policy.compile::<Tap>().unwrap();
            eprintln!("compiled depth 256 on 128 KiB");
            drop(ms);
            eprintln!("dropped compiled output");
        })
        .unwrap()
        .join()
        .unwrap();
    while prefixes.pop().is_some() {}
}

static DROPPED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
static NEXT: Mutex<Option<Weak<Miniscript<Key, Tap>>>> = Mutex::new(None);
static SAW_NEXT: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Key(u32);
impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
}
impl MiniscriptKey for Key {
    type Sha256 = Self;
    type Hash256 = Self;
    type Ripemd160 = Self;
    type Hash160 = Self;
    fn is_x_only_key(&self) -> bool { false }
    fn num_der_paths(&self) -> usize { 0 }
}
impl Drop for Key {
    fn drop(&mut self) {
        if self.0 == 28 {
            SAW_NEXT.store(
                NEXT.lock().unwrap().as_ref().unwrap().upgrade().is_some(),
                Ordering::SeqCst,
            );
        }
        DROPPED.lock().unwrap().push(self.0);
    }
}

#[test]
fn drop_shared_children_and_payloads() {
    type Ms = Miniscript<Key, Tap>;
    fn wrap(node: Terminal<Key, Tap>) -> Ms {
        Ms::from_components_unchecked(node, Ms::TRUE.ty, Ms::TRUE.ext)
    }
    let leaf = |id| Arc::new(Ms::pk_k(Key(id)));
    let shared = leaf(0);
    let weak = Arc::downgrade(&shared);
    let mut deep = wrap(Terminal::Alt(Arc::clone(&shared)));
    for _ in 0..512 {
        deep = wrap(Terminal::Alt(Arc::new(deep)));
    }
    let owner = Arc::new(deep);
    let root = wrap(Terminal::AndV(Arc::clone(&owner), Arc::clone(&owner)));
    drop(root);
    assert_eq!(Arc::strong_count(&owner), 1);
    assert_eq!(Arc::strong_count(&shared), 2);
    assert!(DROPPED.lock().unwrap().is_empty());
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(move || drop(owner))
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(Arc::strong_count(&shared), 1);
    assert!(DROPPED.lock().unwrap().is_empty());
    drop(shared);
    assert!(weak.upgrade().is_none());
    assert_eq!(*DROPPED.lock().unwrap(), vec![0]);

    // Check all recursive variants and normal payload destruction, including order.
    let nodes = vec![
        Terminal::Alt(leaf(1)),
        Terminal::Swap(leaf(2)),
        Terminal::Check(leaf(3)),
        Terminal::DupIf(leaf(4)),
        Terminal::Verify(leaf(5)),
        Terminal::NonZero(leaf(6)),
        Terminal::ZeroNotEqual(leaf(7)),
        Terminal::AndV(leaf(8), leaf(9)),
        Terminal::AndB(leaf(10), leaf(11)),
        Terminal::OrB(leaf(12), leaf(13)),
        Terminal::OrD(leaf(14), leaf(15)),
        Terminal::OrC(leaf(16), leaf(17)),
        Terminal::OrI(leaf(18), leaf(19)),
        Terminal::AndOr(leaf(20), leaf(21), leaf(22)),
        Terminal::Thresh(Threshold::new(1, vec![leaf(23), leaf(24)]).unwrap()),
        Terminal::Sha256(Key(25)),
        Terminal::PkH(Key(26)),
    ];
    for node in nodes {
        drop(wrap(node));
    }
    let node = Ms::pk_k(Key(27)).into_inner();
    assert_eq!(*DROPPED.lock().unwrap(), (0..27).collect::<Vec<_>>());
    drop(node);
    assert_eq!(*DROPPED.lock().unwrap(), (0..28).collect::<Vec<_>>());

    let right = leaf(29);
    *NEXT.lock().unwrap() = Some(Arc::downgrade(&right));
    drop(wrap(Terminal::AndV(leaf(28), right)));
    assert!(SAW_NEXT.load(Ordering::SeqCst));
    assert!(NEXT.lock().unwrap().take().unwrap().upgrade().is_none());
    assert_eq!(*DROPPED.lock().unwrap(), (0..30).collect::<Vec<_>>());
}
