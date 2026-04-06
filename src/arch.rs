//! Architecture-specific ptrace register access
//!
//! Abstracts x86_64 vs aarch64 register differences for syscall tracing.
//! On x86_64: syscall # in orig_rax, args in rdi/rsi/rdx/r10/r8/r9, return in rax, PC in rip
//! On aarch64: syscall # in regs[8], args in regs[0..6], return in regs[0], PC in pc

use anyhow::{Context, Result};
use nix::sys::ptrace;
use nix::unistd::Pid;

/// Architecture-neutral ptrace register snapshot
pub struct PtraceRegs {
    inner: libc::user_regs_struct,
}

impl PtraceRegs {
    /// Read registers from a traced process
    pub fn get(pid: Pid) -> Result<Self> {
        let inner = ptrace::getregs(pid).context("Failed to get registers")?;
        Ok(Self { inner })
    }

    /// Read registers, returning nix::Error on failure (for non-anyhow callers)
    pub fn get_nix(pid: Pid) -> std::result::Result<Self, nix::Error> {
        let inner = ptrace::getregs(pid)?;
        Ok(Self { inner })
    }

    /// Syscall number (orig_rax on x86_64, regs[8] on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn syscall_number(&self) -> i64 {
        self.inner.orig_rax as i64
    }

    #[cfg(target_arch = "aarch64")]
    pub fn syscall_number(&self) -> i64 {
        self.inner.regs[8] as i64
    }

    /// Syscall return value (rax on x86_64, regs[0] on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn syscall_return(&self) -> i64 {
        self.inner.rax as i64
    }

    #[cfg(target_arch = "aarch64")]
    pub fn syscall_return(&self) -> i64 {
        self.inner.regs[0] as i64
    }

    /// Syscall argument 1 (rdi on x86_64, regs[0] on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn arg1(&self) -> u64 {
        self.inner.rdi
    }

    #[cfg(target_arch = "aarch64")]
    pub fn arg1(&self) -> u64 {
        self.inner.regs[0]
    }

    /// Syscall argument 2 (rsi on x86_64, regs[1] on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn arg2(&self) -> u64 {
        self.inner.rsi
    }

    #[cfg(target_arch = "aarch64")]
    pub fn arg2(&self) -> u64 {
        self.inner.regs[1]
    }

    /// Syscall argument 3 (rdx on x86_64, regs[2] on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn arg3(&self) -> u64 {
        self.inner.rdx
    }

    #[cfg(target_arch = "aarch64")]
    pub fn arg3(&self) -> u64 {
        self.inner.regs[2]
    }

    /// Instruction pointer (rip on x86_64, pc on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn instruction_pointer(&self) -> u64 {
        self.inner.rip
    }

    #[cfg(target_arch = "aarch64")]
    pub fn instruction_pointer(&self) -> u64 {
        self.inner.pc
    }

    /// Frame pointer (rbp on x86_64, regs[29] / x29 on aarch64)
    #[cfg(target_arch = "x86_64")]
    pub fn frame_pointer(&self) -> u64 {
        self.inner.rbp
    }

    #[cfg(target_arch = "aarch64")]
    pub fn frame_pointer(&self) -> u64 {
        self.inner.regs[29]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ptrace_regs_struct_exists() {
        // Verify the type compiles on this architecture
        let _size = std::mem::size_of::<libc::user_regs_struct>();
        assert!(_size > 0);
    }
}
