# Subsetix — CUDA interval intersections (1D/2D/3D)

Intersections d'intervalles sur GPU, en CSR par lignes:
- 1D: une seule ligne.
- 2D: lignes = y (offsets par y).
- 3D: lignes aplaties (z,y) avec `row_offsets`, `row_to_y`, `row_to_z`.

Deux passes CUDA (count + write), balayage linéaire par ligne (double pointeur), sorties compactes.

## Schéma de données (CSR)
- Entrées par ensemble S: `begin[]`, `end[]`, `row_offsets[]` (taille rows+1).
- 3D ajoute `row_to_y[]`, `row_to_z[]` (taille rows).
- Résultats (option): indices de ligne (y ou z+y), `r_begin[]`, `r_end[]`, `a_idx[]`, `b_idx[]`.

API (device ptrs): voir `src/interval_intersection.cuh`
- 2D: `findIntervalIntersections(...)` avec offsets par y.
- 3D: `findVolumeIntersections(...)` avec offsets par ligne et maps y/z.
- Réutilisation sans `cudaMalloc`: `compute*IntersectionOffsets(...)` + `write*IntersectionsWithOffsets(...)` (buffers `counts/offsets` fournis par l'appelant, `cudaStream_t` optionnel).

## Build rapide
- Prérequis: CUDA (nvcc), Thrust, CMake, GTest.
- Arch GPU (RTX 1000 Ada): `CMAKE_CUDA_ARCHITECTURES 89` (déjà réglé dans `CMakeLists.txt`).

Commandes:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Exécutables
- `build/intersection_test` — tests 1D/2D/3D (GTest).
- `build/exemple_main` — exemple minimal.
- `build/surface_demo` — 2D (union/VTK désactivés par défaut, voir constantes en tête de fichier).
- `build/volume_demo` — 3D (VTK désactivé par défaut, cf. `WRITE_VTK`).

## Utilisation minimale (2D/3D)
- 2D: fournir `a_begin/a_end/a_y_offsets`, `b_begin/b_end/b_y_offsets`, `y_count` identiques.
- 3D: fournir `*_row_offsets`, `row_to_y`, `row_to_z` (pour A), et `row_count` identiques.
- Les fonctions allouent les buffers résultats (libérer via `freeIntervalResults`/`freeVolumeIntersectionResults`).

Sources utiles:
- `src/interval_intersection.cu` — kernels + orchestration (2 passes, Thrust scan).
- `src/surface_generator.*` / `src/volume_generator.*` — générateurs rect/cercle/box/sphere, union, raster, VTK.
- `src/surface_demo.cu`, `src/volume_demo.cu` — démos + mesures ns/interval.

## Tests et benchmarks
```bash
./build/intersection_test
./build/surface_demo   # ~1M interv./résultat, écrit VTK
./build/volume_demo    # ~1M interv./résultat, VTK off
```
Exemple (RTX 1000 Ada, Release):
- 2D: ~4.9 ns/interval (intersection seule, union/VTK off).
- 3D: ~1.35 ns/interval (intersection seule).

Notes:
- La métrique ns/interval est basée sur le nombre d'intersections produites.
- Adapter les tailles/densités dans les démos (constantes en tête de fichier) pour mesurer à grande échelle.

## Limitations connues
- Scénarios avec très peu de lignes simultanées peuvent sous-utiliser le GPU (un thread = une ligne).
- CMake utilise `find_package(CUDA)` (déprécié) mais fonctionne; modernisation possible.
