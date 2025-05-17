"""
Script to repair the hand_tracking.py file
"""
import re

with open('hand_tracking.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove any unexpected whitespace between methods
content = re.sub(r'(\s+)def\s+', r'\n    def ', content)

# Ensure proper spacing between class methods
content = re.sub(r'(\s+)def', r'\n    def', content)

# Fix any broken docstrings
content = re.sub(r'"""(.*?)"""', lambda m: '"""' + m.group(1).replace('\n',
                 '\n        ') + '"""', content, flags=re.DOTALL)

# Fix indentation in the file
lines = content.split('\n')
fixed_lines = []
in_class = False
class_indent = 0

for line in lines:
    # Skip empty lines
    if not line.strip():
        fixed_lines.append('')
        continue

    # Check if line starts class definition
    if line.startswith('class '):
        in_class = True
        class_indent = 0
        fixed_lines.append(line)
        continue

    # Handle indentation for class methods
    if in_class and line.strip().startswith('def '):
        fixed_lines.append('    ' + line.strip())
    elif in_class:
        # For code inside methods, add 8 spaces (4 for class + 4 for method)
        if line.strip().startswith('#'):
            # Comments inside methods
            fixed_lines.append('        ' + line.strip())
        else:
            # Regular code inside methods
            fixed_lines.append('        ' + line.strip())
    else:
        # Code outside class
        fixed_lines.append(line)

# Write fixed content back to a new file
with open('hand_tracking_fixed.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print("Repair script completed. Check hand_tracking_fixed.py")
