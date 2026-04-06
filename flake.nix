{
  description = "simular - Unified Simulation Engine for the Sovereign AI Stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            wasm-pack
            cargo-tarpaulin
            cargo-criterion
            nodePackages.npm
          ];

          shellHook = ''
            export RUST_BACKTRACE=1
            echo "simular development environment"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
          '';

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "simular";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
        };
      }
    );
}
