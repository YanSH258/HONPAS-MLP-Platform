# HAP_MLP_Project/modules/wrapper.py
import os
from ase.data import atomic_numbers

class InputWrapper:
    def __init__(self, template_path, species_map):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path, 'r') as f:
            self.template_content = f.read()
        self.species_map = species_map

    def _format_blocks(self, frame_data):
        type_names = frame_data['atom_names']
        atom_types = frame_data['atom_types']
        
        # 转换类型索引为原子序数
        all_atom_z = [atomic_numbers[type_names[t_idx]] for t_idx in atom_types]
        
        # 生成 Species Block
        unique_z = sorted(list(set(all_atom_z)))
        species_lines = []
        for z in unique_z:
            if z not in self.species_map:
                raise ValueError(f"Undefined element Z={z} in SPECIES_MAP")
            info = self.species_map[z]
            species_lines.append(f"  {info['index']}   {z}   {info['label']}")
        
        # 生成 Lattice Block
        lattice_lines = [f"  {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}" for row in frame_data['cells'][0]]

        # 生成 Coordinates Block
        coord_lines = []
        for i, coord in enumerate(frame_data['coords'][0]):
            z = all_atom_z[i]
            s_idx = self.species_map[z]['index']
            label = self.species_map[z]['label']
            coord_lines.append(f"  {coord[0]:12.8f}  {coord[1]:12.8f}  {coord[2]:12.8f}   {s_idx}   # {label}")

        return {
            "NumberOfSpecies": len(unique_z),
            "NumberOfAtoms": len(all_atom_z),
            "SpeciesBlock": "\n".join(species_lines),
            "LatticeVectorsBlock": "\n".join(lattice_lines),
            "AtomicCoordinatesBlock": "\n".join(coord_lines)
        }

    def write_input(self, frame_data, output_dir, label="HAP_Task"):
        blocks = self._format_blocks(frame_data)
        final_content = self.template_content.format(SystemLabel=label, **blocks)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, "INPUT.fdf")
        with open(file_path, 'w') as f:
            f.write(final_content)
        return file_path