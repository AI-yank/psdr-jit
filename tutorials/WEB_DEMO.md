# Local GPU demo

From the repository root, with an NVIDIA GPU and a CUDA/OptiX driver:

```sh
python -m pip install ".[web]"
python -m psdr_jit.web
```

Open **http://127.0.0.1:8080**. The first render starts automatically. Change
settings and press **Render** or **Ctrl+Enter** (Cmd+Enter on macOS keyboards).
The first render of a scene or derivative can take longer while kernels compile.
Set **Passes** (`npass`, default 8) and **SPP / pass** to control progressive
rendering. Each pass advances the seed. Both panels display the running mean
of all completed passes, and the footer shows the pass count and total samples
per pixel. The renderer averages linear radiance and signed gradients before
applying exposure or the CubicL color map. **Stop** finishes the current pass
and keeps the accumulated images available to download. **Render** starts a
fresh accumulation with the current controls. Reloading reconnects to an active run.

The demo uses the bundled Cornell-box and environment-lit bunny assets. Choose
path tracing (two bounces) or direct lighting, move the small sphere or bunny,
and adjust roughness, diffuse albedo, or illumination. Position values are
normalized: one unit moves the sphere 100 scene units or the bunny 30 units.

Select **Surface PRB (2)** on the Cornell box or bunny to replay BSDF-sampled
paths for material or emitter derivatives. Small area lights may need more
passes. Use the path or direct integrator for position derivatives.

Choose **Backlit fog** to render a rectangular light behind a bounded
homogeneous medium. The absorption and scattering sliders are in inverse
scene units. Select either coefficient in the gradient panel. The image
includes up to three scattering events. The halo outside the light's outline
comes from in-scattering.

Choose **Homogeneous fog cube** for an angled view of a cube lit by warm and
cool side lights. Set absorption and scattering to zero to see the grid beneath
it, then increase scattering to see the volume appear.

Choose **Smoke plume** for a close view of a heterogeneous vortex with a hollow
core, curling stem, and two colored lights. The medium uses a 28 × 40 × 32
density grid (z, y, x). The sliders scale local absorption and scattering
throughout the grid. Open `/?scene=smoke` to start on this scene.

Open `/?scene=white_room` for **White Room · plume**. Adjust **Density scale**
and press **Render**. Zero removes the plume and one reproduces the validation
scene. **PRB + FD** renders both density gradients from the same samples at each
pass. Both views use one CubicL scale. Relative L2 and RMSE compare the
accumulated linear RGB gradients before color mapping. **Step h** controls the
finite-difference perturbation.

The room uses the published camera, lights and meshes with diffuse materials
and a four-bounce heterogeneous medium. Extract the optional Mitsuba archive
into `tutorials/data` (or your `--assets` directory) so that
`living-room-2/scene_v3.xml`, `models` and `textures` are together in the
`living-room-2` folder. The scene is *The White Room* by Jay-Artist, distributed
by Benedikt Bitterli. [Mitsuba scene archive](https://benedikt-bitterli.me/resources/mitsuba/living-room-2.zip) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

The right panel shows the **signed mean RGB derivative of linear radiance**.
Under **Gradient → Method**, select **Forward AD** or **Finite difference**.
Finite difference keeps the left image at the selected value and varies only
the chosen parameter. Each pass uses the same seed for all three renders.
**Step h** defaults to 0.01 in the slider's units. The estimator uses a central
difference where both offsets fit and a second-order one-sided difference near
a slider limit. For a moving silhouette, increasing the step or the number of
passes can make the change easier to see. The CubicL colorbar runs from purple at the
negative limit through green at zero to peach at the positive limit. Its
symmetric display range is the 99th percentile of absolute derivatives (falling
back to the maximum when sparse), shown numerically under the image. Values
outside this range clip for display. Each render rescales independently, so use
the labels when comparing magnitudes. Exposure affects only the left display.
AD geometry derivatives include primary and secondary boundary samples, each at
the selected sample count. Low sample counts produce noisy estimates.

Rendering runs in one spawned GPU process, with one active job at a time.
Scenes in the same OptiX context share program groups while they are alive.
The C++ renderer clears linked kernels when it creates a new set of groups.
The server binds only to loopback, uses no remote services or CDN, and accepts
only bundled scenes and bounded settings. This is a local demo, not a public
multi-user render service. Stop it with Ctrl+C; an active GPU job finishes before
shutdown. To use another port or launch outside the checkout:

```sh
python -m psdr_jit.web --port 8081 --assets /path/to/psdr-jit/tutorials/data
```

The wheel includes the web UI; scene assets remain in the repository. No Node.js
build step is needed. Install `.[web,test]` and run `python -m pytest -q` to check
scene derivatives, illumination linearity, input validation, and the HTTP-to-GPU
pipeline. GPU checks require CUDA. The webpage should also be checked in a browser
after changes to its controls or layout.
