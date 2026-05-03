import re
import sys

def join_wrapped_lines(text):
    # Split into paragraphs on blank lines, then join wrapped lines within each
    paragraphs = re.split(r'\n\s*\n', text.strip())
    joined = []
    for para in paragraphs:
        # Join lines within paragraph into one line
        single = ' '.join(line.strip() for line in para.splitlines() if line.strip())
        joined.append(single)
    return '\n\n'.join(joined)

if len(sys.argv) < 2:
    print("Usage: python join_lines.py input.txt [output.txt]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

result = join_wrapped_lines(content)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(result)

print(f"Done! Saved to {output_file}")
