// SPDX-License-Identifier: CC0-1.0

//! Flat policies and compilation dependencies for the iterative compiler.

use core::num::NonZeroU32;

use crate::iter::TreeLike;
use crate::policy::Concrete;
use crate::prelude::*;
use crate::{MiniscriptKey, PositiveF64, Threshold};

/// Only atomic policies are borrowed. Children are indices, so comparing nodes and
/// constructing the conjunctions used for n-of-n thresholds do not recurse.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum PolicyNode<'a, Pk: MiniscriptKey> {
    Atom(&'a Concrete<Pk>),
    And([usize; 2]),
    Or([(NonZeroU32, usize); 2]),
    Thresh {
        children: Threshold<usize, 0>,
        conjunction: Option<usize>,
    },
}

pub(super) fn flatten_policy<Pk: MiniscriptKey>(
    policy: &Concrete<Pk>,
) -> (Vec<PolicyNode<'_, Pk>>, usize) {
    fn intern<'a, Pk: MiniscriptKey>(
        nodes: &mut Vec<PolicyNode<'a, Pk>>,
        indices: &mut BTreeMap<PolicyNode<'a, Pk>, usize>,
        node: PolicyNode<'a, Pk>,
    ) -> usize {
        if let Some(index) = indices.get(&node) {
            return *index;
        }
        let index = nodes.len();
        nodes.push(node.clone());
        indices.insert(node, index);
        index
    }

    let mut nodes = Vec::new();
    let mut indices = BTreeMap::new();
    let mut stack = Vec::new();
    // Reverse preorder visits children before their parent, with the leftmost child
    // on top of the stack. The preorder iterator itself uses an explicit stack.
    for policy in policy
        .pre_order_iter()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let node = match policy {
            Concrete::And(subs) => {
                assert_eq!(subs.len(), 2, "and takes 2 args");
                PolicyNode::And([stack.pop().unwrap(), stack.pop().unwrap()])
            }
            Concrete::Or(subs) => {
                assert_eq!(subs.len(), 2, "or takes 2 args");
                PolicyNode::Or([
                    (subs[0].0, stack.pop().unwrap()),
                    (subs[1].0, stack.pop().unwrap()),
                ])
            }
            Concrete::Thresh(thresh) => {
                let children = thresh.map_ref(|_| stack.pop().unwrap());
                let conjunction = if children.is_and() {
                    let mut iter = children.iter().copied();
                    let mut left = iter.next().expect("thresholds are nonempty");
                    for right in iter {
                        left = intern(&mut nodes, &mut indices, PolicyNode::And([left, right]));
                    }
                    Some(left)
                } else {
                    None
                };
                PolicyNode::Thresh { children, conjunction }
            }
            atom => PolicyNode::Atom(atom),
        };
        stack.push(intern(&mut nodes, &mut indices, node));
    }
    assert_eq!(stack.len(), 1);
    (nodes, stack.pop().unwrap())
}

/// A policy may need several compilations with different satisfaction costs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CompilationTask {
    pub policy: usize,
    pub sat_prob: PositiveF64,
    pub dissat_prob: Option<PositiveF64>,
}

impl CompilationTask {
    pub fn new(policy: usize, sat_prob: PositiveF64, dissat_prob: Option<PositiveF64>) -> Self {
        Self { policy, sat_prob, dissat_prob }
    }

    /// Visit dependencies in evaluation order. A task with a dissatisfaction cost
    /// first compiles the same policy without one, including its dependencies.
    pub fn for_each_dependency<Pk: MiniscriptKey>(
        self,
        nodes: &[PolicyNode<'_, Pk>],
        mut visit: impl FnMut(Self),
    ) {
        let Self { policy, sat_prob, dissat_prob } = self;
        if dissat_prob.is_some() {
            visit(Self::new(policy, sat_prob, None));
        }
        let mut add = |policy, sp, dp| visit(Self::new(policy, sp, dp));
        match &nodes[policy] {
            PolicyNode::Atom(_) => {}
            PolicyNode::And(subs) => {
                for child in subs {
                    add(*child, sat_prob, dissat_prob);
                }
            }
            PolicyNode::Or(subs) => {
                let (lw, rw) = or_weights(subs);
                // andor candidates use the grandchildren of either And branch.
                for (branch, weight, other_weight) in [(subs[0].1, lw, rw), (subs[1].1, rw, lw)] {
                    if let PolicyNode::And(children) = &nodes[branch] {
                        for child in children {
                            add(
                                *child,
                                weight * sat_prob,
                                Some((other_weight * sat_prob).conditional_add(dissat_prob)),
                            );
                        }
                    }
                }
                for (child, weight, other_weight) in [(subs[0].1, lw, rw), (subs[1].1, rw, lw)] {
                    // Each Some task also compiles None, so it need not be queued separately.
                    for dp in or_dissat_probs(other_weight, sat_prob, dissat_prob)
                        .into_iter()
                        .flatten()
                    {
                        add(child, weight * sat_prob, Some(dp));
                    }
                }
            }
            PolicyNode::Thresh { children, conjunction } => {
                let (sp, dp) = threshold_probs(children, sat_prob, dissat_prob);
                for child in children.iter() {
                    add(*child, sp, dp);
                }
                if let Some(conjunction) = conjunction {
                    add(*conjunction, sat_prob, dissat_prob);
                }
            }
        }
    }
}

pub(super) fn or_weights(subs: &[(NonZeroU32, usize); 2]) -> (PositiveF64, PositiveF64) {
    let total = PositiveF64::from(subs[0].0) + PositiveF64::from(subs[1].0);
    (PositiveF64::from(subs[0].0) / total, PositiveF64::from(subs[1].0) / total)
}

pub(super) fn or_dissat_probs(
    weight: PositiveF64,
    sat_prob: PositiveF64,
    dissat_prob: Option<PositiveF64>,
) -> [Option<PositiveF64>; 4] {
    [
        Some((weight * sat_prob).conditional_add(dissat_prob)),
        Some(weight * sat_prob),
        dissat_prob,
        None,
    ]
}

pub(super) fn threshold_probs(
    thresh: &Threshold<usize, 0>,
    sat_prob: PositiveF64,
    dissat_prob: Option<PositiveF64>,
) -> (PositiveF64, Option<PositiveF64>) {
    let sp = sat_prob * PositiveF64::k_over_n(thresh);
    let dp = match (dissat_prob, PositiveF64::one_minus_k_over_n(thresh)) {
        (Some(dp), Some(kn)) => Some(dp + kn * sat_prob),
        (Some(dp), None) => Some(dp),
        (None, Some(kn)) => Some(kn * sat_prob),
        (None, None) => None,
    };
    (sp, dp)
}
