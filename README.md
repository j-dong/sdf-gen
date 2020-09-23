SDF generator
=============

Generates signed distance fields from an SVG file.
Also produces very accurate gradients.
Output currently has a normal map in the RGB components and the SDF
in the alpha component. This can be easily changed in `main.rs`.

Limitations
-----------

* Only supports a single path. Fixing this would require adding
  some kind of boolean path manipulation code, so that's probably
  never going to happen.
* Paths must be compatible with *both* `even-odd` and `nonzero` fill
  modes. This is because when I was writing the GPU renderer I assumed
  `even-odd`, but I had to switch to `nonzero` on the CPU as it's more
  robust. This can be fixed pretty easily, but I don't feel like it.
* All paths are stored as *quadratic* bezier segments. This is a
  fundamental limitation and can't be fixed without significantly
  complicating the code.

Why should you use this library?
--------------------------------

* It's pretty good at handling complex paths.
* It's decently fast.
* It generates accurate gradients.
* SDF is generated based on curves rather than based on rasterization.
