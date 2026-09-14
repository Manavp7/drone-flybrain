# Mantis walking actor

The bundled **Cesium Man** model is © 2017 **Cesium**, licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The complete license is in [CC-BY-4.0.txt](CC-BY-4.0.txt).

Source: [Khronos glTF Sample Assets — Cesium Man](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/90d7ede14c7e280af263824604b427a1ca02cb66/Models/CesiumMan),
revision `90d7ede14c7e280af263824604b427a1ca02cb66`.
The bundled `CesiumMan.glb` is unchanged from that revision. Its SHA-256 is
`b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9`.
Original asset license metadata is retained in `UPSTREAM-METADATA.json`.

Mantis adapts this model at runtime: it converts coordinates to the simulator's
Z-up convention, normalizes initial-pose height to 1.8 m, maps the existing 19
animated joints to MuJoCo mocap bones and blends the original mesh skin. The
default appearance assigns plain blue shirt, dark trousers, brown skin and dark
shoes by fixed skeletal influences. It replaces the source logo texture with
plain materials; geometry and the two-second walking animation are retained.
These are simulation actors with stylized proportions, not scans of real people.

The original GLB also contains a © 2015 Cesium logo with a separate trademark
notice, retained verbatim in `Cesium-Trademark.txt`. The default clothing
appearance does not render that logo. The optional `appearance="source"` mode
retains the source artwork; the asset license does not grant additional trademark
rights. No endorsement by Cesium or Khronos is implied.

Images and videos showing the adapted character must retain this Cesium credit,
CC BY 4.0 license link and a statement that the character was adapted as above.
Other scene elements, code and earlier photo-based footage have their own notices.
