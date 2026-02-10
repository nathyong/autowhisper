{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenvNoCC.mkDerivation {
  name = "shell";
  nativeBuildInputs = [
    pkgs.uv
    pkgs.python3Packages.numpy
    pkgs.sox
    pkgs.libnotify
  ];
}
