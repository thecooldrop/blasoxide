use std::alloc;

pub struct Alloc {
    ptr: *mut u8,
    layout: alloc::Layout,
}

impl Alloc {
    pub fn new(size: usize) -> Alloc {
        const ALIGN: usize = 64;
        let layout = alloc::Layout::from_size_align(size, ALIGN).unwrap();
        let ptr = unsafe { alloc::alloc(layout) };
        Alloc { ptr, layout }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for Alloc {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr, self.layout);
        }
    }
}
