// SPDX-License-Identifier: CC0-1.0

//! Ownership of temporary Miniscripts in the compiler.

use core::ops::Deref;

use crate::prelude::sync::Arc;
use crate::prelude::Vec;
use crate::{Miniscript, MiniscriptKey, ScriptContext, Terminal};

/// Owns a temporary Miniscript and drops its unshared descendants iteratively.
#[derive(Clone, Debug)]
pub(super) struct CompilerMiniscript<Pk: MiniscriptKey, Ctx: ScriptContext>(
    Option<Arc<Miniscript<Pk, Ctx>>>,
);

impl<Pk: MiniscriptKey, Ctx: ScriptContext> CompilerMiniscript<Pk, Ctx> {
    pub fn new(ms: Miniscript<Pk, Ctx>) -> Self { Self(Some(Arc::new(ms))) }

    pub fn into_inner(mut self) -> Miniscript<Pk, Ctx> {
        Arc::try_unwrap(self.0.take().expect("compiler Miniscript is present"))
            .expect("only the selected compilation remains")
    }
}

impl<Pk: MiniscriptKey, Ctx: ScriptContext> Deref for CompilerMiniscript<Pk, Ctx> {
    type Target = Arc<Miniscript<Pk, Ctx>>;

    fn deref(&self) -> &Self::Target { self.0.as_ref().expect("compiler Miniscript is present") }
}

impl<Pk: MiniscriptKey, Ctx: ScriptContext> Drop for CompilerMiniscript<Pk, Ctx> {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        let mut next = self.0.take();
        while let Some(child) = next {
            // Shared children stay owned by their remaining references.
            if let Ok(ms) = Arc::try_unwrap(child) {
                match ms.node {
                    Terminal::Alt(child)
                    | Terminal::Swap(child)
                    | Terminal::Check(child)
                    | Terminal::DupIf(child)
                    | Terminal::Verify(child)
                    | Terminal::NonZero(child)
                    | Terminal::ZeroNotEqual(child) => pending.push(child),
                    Terminal::AndV(left, right)
                    | Terminal::AndB(left, right)
                    | Terminal::OrB(left, right)
                    | Terminal::OrD(left, right)
                    | Terminal::OrC(left, right)
                    | Terminal::OrI(left, right) => pending.extend([right, left]),
                    Terminal::AndOr(a, b, c) => pending.extend([c, b, a]),
                    Terminal::Thresh(thresh) => {
                        pending.extend(thresh.into_data().into_iter().rev())
                    }
                    leaf => drop(leaf),
                }
            }
            next = pending.pop();
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use core::cmp::Ordering;
    use core::fmt;
    use core::hash::{Hash, Hasher};
    use std::sync::{Mutex, Weak};

    use super::*;
    use crate::{Tap, Threshold};

    #[derive(Debug, Default)]
    struct DropState {
        dropped: Vec<u32>,
        next: Option<Weak<Miniscript<Key, Tap>>>,
        saw_next: bool,
    }

    #[derive(Clone, Debug)]
    struct Key(u32, Arc<Mutex<DropState>>);
    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    }
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> Ordering { self.0.cmp(&other.0) }
    }
    impl Hash for Key {
        fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
    }
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
            let mut state = self.1.lock().unwrap();
            if self.0 == 28 {
                state.saw_next = state.next.as_ref().unwrap().upgrade().is_some();
            }
            state.dropped.push(self.0);
        }
    }

    #[test]
    fn drop_shared_children_and_payloads() {
        type Ms = Miniscript<Key, Tap>;
        fn wrap(node: Terminal<Key, Tap>) -> Ms {
            Ms::from_components_unchecked(node, Ms::TRUE.ty, Ms::TRUE.ext)
        }
        let state = Arc::new(Mutex::new(DropState::default()));
        let key = |id| Key(id, Arc::clone(&state));
        let leaf = |id| Arc::new(Ms::pk_k(key(id)));
        let shared = leaf(0);
        let weak = Arc::downgrade(&shared);
        let mut deep = wrap(Terminal::Alt(Arc::clone(&shared)));
        for _ in 0..512 {
            deep = wrap(Terminal::Alt(Arc::new(deep)));
        }
        let owner = Arc::new(deep);
        let root = wrap(Terminal::AndV(Arc::clone(&owner), Arc::clone(&owner)));
        drop(CompilerMiniscript::new(root));
        assert_eq!(Arc::strong_count(&owner), 1);
        assert_eq!(Arc::strong_count(&shared), 2);
        assert!(state.lock().unwrap().dropped.is_empty());
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(move || drop(CompilerMiniscript(Some(owner))))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(Arc::strong_count(&shared), 1);
        assert!(state.lock().unwrap().dropped.is_empty());
        drop(shared);
        assert!(weak.upgrade().is_none());
        assert_eq!(state.lock().unwrap().dropped, vec![0]);

        // Each key should be dropped once, in left-to-right child order.
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
            Terminal::Sha256(key(25)),
            Terminal::PkH(key(26)),
        ];
        for node in nodes {
            drop(CompilerMiniscript::new(wrap(node)));
        }
        let node = CompilerMiniscript::new(Ms::pk_k(key(27))).into_inner().node;
        assert_eq!(state.lock().unwrap().dropped, (0..27).collect::<Vec<_>>());
        drop(node);
        assert_eq!(state.lock().unwrap().dropped, (0..28).collect::<Vec<_>>());

        let right = leaf(29);
        state.lock().unwrap().next = Some(Arc::downgrade(&right));
        drop(CompilerMiniscript::new(wrap(Terminal::AndV(leaf(28), right))));
        let state = state.lock().unwrap();
        assert!(state.saw_next);
        assert!(state.next.as_ref().unwrap().upgrade().is_none());
        assert_eq!(state.dropped, (0..30).collect::<Vec<_>>());
    }
}
