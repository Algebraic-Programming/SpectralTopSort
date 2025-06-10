# rMLGP needs one edge per line, and the nodes declrared at start.
# This script modifies the file if multiple edges per line are detected.
# Example correct format for rMLGP:
# digraph G {
# 0;
# 1;
# 2;
# 3;
# 4;
# 0->1 ;
# 0->2 ;
# 0->4 ;
# 1->5 ;
# 2->3 ;
# 3->4 ;
# }

import sys

def convert_dot_file(input_file, output_file):
    nodes = set()
    edges = []
    
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('//', '#')):
                continue
            if (line.find('graph') != -1) and (line.find('name') != -1):
                continue
            
            # Find nodes
            if line.endswith(';') and '->' not in line:
                node = line[:-1].strip()
                nodes.add(node)
                continue
            
            # Find edges
            if '->' in line:
                parts = line.split('->')
                src = parts[0].strip()
                targets_part = parts[1].replace('{', '').replace('}', '').replace(';', '')
                
                for target in targets_part.split():
                    if target.find("[") != -1:
                        continue
                    edges.append(f"{src}->{target} ;")
                    nodes.update([src, target])
    
    # Write converted file
    with open(output_file, 'w') as f:
        f.write("digraph G {\n")
        for node in sorted(nodes, key=int):
            f.write(f"{node};\n")
        for edge in edges:
            f.write(f"{edge}\n")
        f.write("}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_for_dagP.py input.dot output.dot")
        sys.exit(1)
    
    convert_dot_file(sys.argv[1], sys.argv[2])
    