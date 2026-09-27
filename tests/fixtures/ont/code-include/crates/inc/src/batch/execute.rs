// The impl lives here and names its type through `use super::…`: the walk must find the method in THIS
// module, not follow the `use` back to where the type is defined.
use super::Batch;

impl Batch {
    pub fn execute(&self) {}
}
