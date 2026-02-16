{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenvNoCC.mkDerivation {
  name = "shell";
  nativeBuildInputs = [
    pkgs.uv
    pkgs.python3Packages.numpy
    pkgs.libnotify
    pkgs.pipewire
  ];
}
