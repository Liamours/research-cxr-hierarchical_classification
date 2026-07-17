# C. Label hierarchy

The 51 canonical findings are drawn from Indonesian clinical guidelines
(PDPI, PNPK, KKI) and represent the full scope of thoracic findings relevant
to the target clinical deployment context, independent of which findings
current public datasets happen to annotate. Because public datasets annotate
radiographic findings while the canonical set targets clinical diagnoses,
only 27 of the 51 labels currently receive training signal from the
implemented dataset adapters (Section B); the remaining 24 are retained in
the label space and supervised as soon as a dataset provides them.

For the hierarchical training and inference modes, we define 13 IS-A edges
over the 51 labels, grouping more specific findings under clinically broader
parents (src/data/hierarchy.py): Pneumonia is parent of COVID-19 Pneumonia,
Aspiration Pneumonia, and Other Viral Pneumonia; ILD is parent of IPF, COP,
Hypersensitivity Pneumonitis, Silicosis, Asbestosis, Other Pneumoconiosis, and
Sarcoidosis; Tuberculosis is parent of Post-TB Obstructive Syndrome; Pleural
Effusion is parent of Pleural Empyema; Pulmonary Hypertension is parent of Cor
Pulmonale. These edges are clinically grounded IS-A relationships and are not
an official published taxonomy. The hierarchy is supplied as an external
mapping rather than hard-coded, so a different edge set can be used without
changing code.
