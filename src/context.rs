use threadpool::ThreadPool;
use crate::aligned_alloc::Alloc;
use std::mem;

pub struct Context {
	f32_blocksizes: BlockSizes,
	f64_blocksizes: BlockSizes,
	thread_pool: ThreadPool,
	pa: Alloc,
	pb: Alloc,
}

struct BlockSizes {
	mc: usize,
	kc: usize,
	nc: usize,
}

impl Context {
	pub fn new() -> Context {
		let f32_blocksizes = BlockSizes {
			mc: 256,
			kc: 512,
			nc: 4096,
		};

		let f64_blocksizes = BlockSizes {
			mc: f32_blocksizes.mc / 2,
			kc: f32_blocksizes.kc,
			nc: f32_blocksizes.nc,
		};

		let thread_pool = threadpool::Builder::new().build();

		let pa = Alloc::new(f32_blocksizes.mc * f32_blocksizes.kc * mem::size_of::<f32>());

		let pb = Alloc::new(f32_blocksizes.kc * f32_blocksizes.nc * mem::size_of::<f32>());

		Context {
			f32_blocksizes,
			f64_blocksizes,
			thread_pool,
			pa,
			pb,
		}
	}

	pub(crate) fn smc(&self) -> usize {
		self.f32_blocksizes.mc
	}

	pub(crate) fn skc(&self) -> usize {
		self.f32_blocksizes.kc
	}

	pub(crate) fn snc(&self) -> usize {
		self.f32_blocksizes.nc
	}

	pub(crate) fn dmc(&self) -> usize {
		self.f64_blocksizes.mc
	}

	pub(crate) fn dkc(&self) -> usize {
		self.f64_blocksizes.kc
	}

	pub(crate) fn dnc(&self) -> usize {
		self.f64_blocksizes.nc
	}

	pub(crate) fn spa(&self) -> *mut f32 {
		self.pa.ptr() as *mut f32
	}

	pub(crate) fn spb(&self) -> *mut f32 {
		self.pb.ptr() as *mut f32
	}

	pub(crate) fn dpa(&self) -> *mut f64 {
		self.pa.ptr() as *mut f64
	}

	pub(crate) fn dpb(&self) -> *mut f64 {
		self.pb.ptr() as *mut f64
	}

	pub(crate) fn execute<F: FnOnce(usize) + Send + 'static + Copy>(&self, start: usize, end: usize, step: usize, f: F) {
		let len = end - start;
		let num_steps = len / step;
		let thread_count = self.thread_pool.max_count();
		let left_steps = num_steps % thread_count;
		let main_steps = num_steps - left_steps;

		let job_size = main_steps / thread_count;

		if job_size > 0 {
			for j in (0..main_steps).step_by(job_size) {
				self.thread_pool.execute(move || {
					for i in j..j+job_size {
						f(start + i * step);
					}
				});
			}
		}


		for i in main_steps..num_steps {
			f(start + i * step);
		}

		self.thread_pool.join();
	}
}
