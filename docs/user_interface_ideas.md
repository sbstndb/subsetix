# User-Oriented API (HPC/AMR context)

Objectif: éviter la micromanipulation de buffers, exposer un pipeline CUDA moderne, asynchrone et réutilisable.

## Douleurs actuelles
- L'utilisateur doit fabriquer manuellement les tableaux CSR, malloc/copy/free.
- Aucun contrôle de flux (streams/graphes), difficile d'intégrer dans une boucle AMR.
- Impossibilité de réutiliser un plan de calcul à taille quasi-fixe.

## Couches proposées
1. **HostIntervalSet** (1D/2D/3D)
   - Constructeurs explicites (`fromIntervals`, `fromSurface`, `fromVolume`).
   - Compile la hiérarchie AMR: niveaux ➝ vecteurs CSR concaténés + offsets de niveau.
2. **DeviceIntervalSet**
   - `upload(stream)` avec mémoire épinglée + `cudaMallocAsync` via `cudaMemPool_t`.
   - RAII, expose des vues `device_span<int>` pour intégration custom.
3. **IntersectionPlan**
   - Configure threads/blocs, buffers résultats, et graphe CUDA.
   - `record(stream)` enregistre count+scan+write (graph capture) pour relance low-overhead.
   - Possibilité d'activer un mode « streaming double buffer » (AMR time-stepping).

## API escomptée (esquisse)
```cpp
IntersectionContext ctx{.stream = user_stream, .mempool = my_pool};
HostIntervalSet a = HostIntervalSet::fromVolume(volA, Layout::ZY_CSR);
HostIntervalSet b = HostIntervalSet::fromVolume(volB, Layout::ZY_CSR);
DeviceIntervalSet dA = ctx.upload(a);
DeviceIntervalSet dB = ctx.upload(b);

IntersectionPlan plan = ctx.make_plan(dA, dB, Options{
    .levels = amr_levels,
    .output_format = OutputFormat::RowMajor,
    .keep_indices = true
});

ResultHandle res = plan.execute_async();
res.copy_to_host_async(host_buffer);
```

## CUDA features à exploiter
- **Streams multiples**: permettre un stream par niveau AMR.
- **CUDA graphs**: capture des deux kernels + scan pour éliminer le coût de lancement.
- **cudaMallocAsync/mempools**: amortir les allocs de buffers temporaires.
- **Events/timestamps**: exposer des métriques ns/interval prêts pour profiling.
- **Device-side memcpy**: version « in-place » pour intersections successives.

## AMR & flexibilité
- Supporter `level_offsets[]` (CSR concaténée) pour passer du niveau à la ligne physique.
- Option pour renvoyer uniquement des stats (count) pour raffinement décisionnel.
- Hooks pour fournir des `thrust::device_span<const int>` au lieu de copies.

## Étapes
1. Implémenter Host/DeviceIntervalSet + gestion mempool.
2. Refactoriser les kernels pour accepter un mapping pré-calculé `interval_to_row`.
3. Introduire IntersectionPlan avec capture de graph + exécution asynchrone.
4. Ajouter tests d'intégration (AMR 2 niveaux) + benchmarks stream/graph.
5. Documenter un guide « intégration AMR ».
