{
  description = "anduril — minimal agent for OpenAI-compatible endpoints";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Core (anduril) Python — matches the "one runtime dep" promise.
        python = pkgs.python3.withPackages (ps: [
          ps.openai
          ps.rich
          ps.pylatexenc
        ]);
        # Dev shell adds lint, test, and the optional web-skill deps.
        pythonDev = pkgs.python3.withPackages (ps: [
          ps.openai
          ps.rich
          ps.pylatexenc
          ps.ruff
          ps.pytest
          # Optional web-skill deps. Drop if you don't use the web skill.
          ps.ddgs
          ps.httpx
          ps.trafilatura
        ]);
      in
      {
        # Minimal shell matching the runtime promise (openai only).
        devShells.minimal = pkgs.mkShell {
          packages = [ python pkgs.uv ];
        };
        # Full dev shell with lint, test, and the web skill.
        devShells.default = pkgs.mkShell {
          packages = [ pythonDev pkgs.uv ];
        };
      }
    );
}
