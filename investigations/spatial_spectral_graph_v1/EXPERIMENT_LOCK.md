# Single-step unlabeled target graph probe

This is an exploratory frozen-prediction diagnostic, not a SpectralFlow model claim. The current Houston18 GT has been inspected during prior development. Before scoring this probe, lock one rule and apply it to **all** of A raw, A + existing fixed HyperSIGMA soft-KL `lambda=1`, full MLUDA raw, and full MLUDA + the same fixed prior at seeds 202601/202602/202603. No checkpoint, model weight, teacher prior, classifier, Flow component, or metric definition changes.

Construct a graph on the 53,184 officially predicted target centers only, preserving their official order. Connect 4-neighbor spatial centers when both occur in this set. Use the existing target ILDA 48-band center spectrum, L2 normalized, and a nonnegative edge weight `w_ij = exp(-(1-cosine(x_i,x_j))/median_edge_distance)`, where the median is computed from **all unlabeled graph edges** in that seed. If the median is zero, use the smallest positive machine epsilon. No target class values enter graph construction. Perform exactly one simultaneous Jacobi step:

`q_graph(i) = 0.5*q(i) + 0.5*sum_j(w_ij*q(j))/sum_j(w_ij)`

For isolated centers, keep `q(i)`. The coefficient 0.5, 4-neighbor graph, ILDA spectrum, and one step are fixed implementation choices, not tuned on target GT. Save all graph predictions plus hashes and graph diagnostics before opening GT. Post-hoc compare OA (official `correct/53,200`), AA/Kappa (53,184 predictions), all class recalls, class mass, and candidate coverage/purity. Stop if OA improves only by collapsing minority classes. Any positive result requires new seeds or another scene for confirmation, and spatial graph postprocessing itself is a baseline rather than a novel Flow contribution.
