"""The in-context design loop from the AIDO Cell report, run against luria-aido.

Verified to run end to end (16 steps, 0 failures). Note the one place it differs
from the published snippet: `design_molecule_for_target` returns a provenance
dict, so the SMILES has to be unwrapped before it is handed to a perturbation.

What the readouts are worth here:
  get_protein_structure / get_protein_ligand_interactions   delegated, real
  get_expression                                            F2 — does not read
                                                            the molecule; see
                                                            MODEL_CARD.md
"""
from luria_aido import Cell

N = 2
IMATINIB = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"


def main() -> None:
    cell = Cell("K-562")

    ref = cell.clone()
    ref.small_molecule_perturbation(IMATINIB)
    ref_expr = ref.get_expression()
    print(f"reference expression: {ref_expr.shape[0]} genes")

    for i in range(N):
        proposal = cell.design_molecule_for_target("ABL1")
        valid = proposal.get("value") or []
        if not valid:
            print(f"[{i}] generator returned no valid SMILES")
            continue
        smiles = valid[0]["smiles"]

        cp = cell.clone()
        cp.small_molecule_perturbation(smiles)
        structure = cp.get_protein_structure("ABL1", smiles)
        contacts = cp.get_protein_ligand_interactions(smiles)
        expr = cp.get_expression()

        print(
            f"[{i}] {smiles[:52]}\n"
            f"     structure={structure.get('source', '?')}  "
            f"contacts={contacts.get('source', '?')}  "
            f"expression={expr.shape[0]} genes"
        )


if __name__ == "__main__":
    main()
