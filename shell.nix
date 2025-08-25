{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenvNoCC.mkDerivation {
  name = "shell";
  nativeBuildInputs = [
    pkgs.uv
    pkgs.sox
  ];
}
