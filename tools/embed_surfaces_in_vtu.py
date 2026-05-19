"""
Lê um VTU com tetraedros tagueados por region_id, calcula as faces de
fronteira de cada região e exporta um VTU contendo apenas os triângulos
de superfície, com o region_id correspondente.

Uso:
     python3 embed_surfaces_in_vtu.py \
        -i fenicsFiles/Patient_2.vtu \
        -o Patient_2_surfaces.vtu

    # Campo de tag se diferente de region_id:
     python3 embed_surfaces_in_vtu.py \
        -i Patient.vtu -o out.vtu --tag_field meu_campo
"""

import argparse
import numpy as np
import meshio


def embed_surfaces(vtu_path, output_path, region_id_key="region_id"):
    mesh = meshio.read(vtu_path)

    tetra_block = next((c for c in mesh.cells if c.type == "tetra"), None)
    if tetra_block is None:
        raise ValueError("Nenhum bloco de tetraedros encontrado no VTU.")

    tetras  = tetra_block.data
    reg_ids = mesh.cell_data[region_id_key][0].astype(int).flatten()

    face_combos = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])

    # Gera todas as faces de todos os tetras, rastreando de qual tetra vieram
    all_faces     = tetras[:, face_combos].reshape(-1, 3)          # (4N, 3)
    tetra_of_face = np.repeat(np.arange(len(tetras)), 4)           # (4N,)

    # Faces externas: aparecem só uma vez em toda a malha
    faces_sorted = np.sort(all_faces, axis=1)
    _, idx, counts = np.unique(faces_sorted, axis=0,
                               return_index=True, return_counts=True)
    outer_idx   = idx[counts == 1]
    tri_faces   = all_faces[outer_idx]
    tri_tags    = reg_ids[tetra_of_face[outer_idx]]

    for tag in np.unique(tri_tags):
        print(f"  Região {tag}: {(tri_tags == tag).sum()} triângulos")

    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[meshio.CellBlock("triangle", tri_faces)],
        cell_data={region_id_key: [tri_tags.astype(float)]},
    )
    meshio.write(output_path, out_mesh)
    print(f"[OK] Salvo: {output_path}  ({len(tri_faces)} triângulos no total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embutir superfícies de fronteira em VTU.")
    parser.add_argument("-i", "--input",     required=True,              help="VTU de entrada")
    parser.add_argument("-o", "--output",    required=True,              help="VTU de saída")
    parser.add_argument("--tag_field",       default="region_id",        help="Campo de cell_data com as tags de região (padrão: region_id)")
    args = parser.parse_args()

    embed_surfaces(args.input, args.output, region_id_key=args.tag_field)
