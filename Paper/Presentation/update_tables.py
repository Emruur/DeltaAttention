import re

with open('index.html', 'r') as f:
    content = f.read()

# First replace step3-target with step2-target in the bottom right table
start_idx = content.find('<div class="table-title">Aggregation @128K (LLaMA)</div>')
end_idx = content.find('</section>', start_idx)

if start_idx != -1 and end_idx != -1:
    sub_content = content[start_idx:end_idx]
    sub_content = sub_content.replace('step3-target', 'step2-target')
    content = content[:start_idx] + sub_content + content[end_idx:]

# Now replace the <td>...</td> for the 11 tasks in the bottom left table
# We only want to target lines between:
start_idx2 = content.find('RULER (Full) per-task accuracy')
end_idx2 = content.find('</table>', start_idx2)

if start_idx2 != -1 and end_idx2 != -1:
    sub_content = content[start_idx2:end_idx2]
    # Function to replace td and th that don't have step2-target (which are the 11 tasks)
    # The columns are 1 to 8 and 11 to 13, but actually we can just match <th> and <td> that contain numbers or text, excluding the ones with class
    
    def replacer(match):
        tag = match.group(1) # 'th' or 'td'
        inner = match.group(2)
        if 'class=' in match.group(0):
            return match.group(0) # skip already classed ones like Method or vt/cwe
        if inner == 'Method':
            return match.group(0)
        return f'<{tag} class="step3-target">{inner}</{tag}>'
    
    sub_content = re.sub(r'<(th|td)>([^<]+)</\1>', replacer, sub_content)
    
    content = content[:start_idx2] + sub_content + content[end_idx2:]

with open('index.html', 'w') as f:
    f.write(content)
print("Updated index.html")
