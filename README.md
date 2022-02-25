# Wari
Experimental WebAssembly Runtime for RISC processors.

Initially, wari will only target RISC-V.  In the future, other processors may be
targeted.

## Runtime Stages
 1. Load module with
    [`parity_wasm::elements::Module::from_bytes()`](https://docs.rs/parity-wasm/0.42.2/parity_wasm/elements/struct.Module.html#method.from_bytes)
 2. Get individual module sections
 3. Pass each section into a converter to RISC instructions
 4. Optimize instructions based on known patterns that can be simplified

Steps 1 & 2 can happen at the same time (loading function) as well as 3 & 4
(with a peekable iterator).

## Testing With QEMU
TODO
