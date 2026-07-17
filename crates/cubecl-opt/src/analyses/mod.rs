mod base;
pub mod dominance;
// The range analysis is implemented for index-bound optimization but is not
// wired into the active pass pipeline yet.
#[allow(dead_code)]
pub mod integer_range;
pub mod liveness;
pub mod post_order;
pub mod uniformity;
pub mod writes;

pub use base::*;
