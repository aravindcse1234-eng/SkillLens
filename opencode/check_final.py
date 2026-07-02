import ast
src = open(r'C:\Data Science Projects\SkillLensAI\app.py', encoding='utf-8').read()
try:
    ast.parse(src)
    print('Syntax: OK')
except SyntaxError as e:
    print('Syntax error: line {}: {}'.format(e.lineno, e.msg))

checks = {
    'HIGH SIGHT text': 'HIGH SIGHT' in src,
    'Command Center text': 'Command Center' in src,
    'D4AF37 gold in CSS': '#D4AF37' in src and 'st.markdown' in src[src.find('#D4AF37')-200:src.find('#D4AF37')+200] and '<style>' in src[src.find('#D4AF37')-400:src.find('#D4AF37')],
    '00F5D4 teal in CSS': '#00F5D4' in src and 'st.markdown' in src[src.find('#00F5D4')-200:src.find('#00F5D4')+200] and '<style>' in src[src.find('#00F5D4')-400:src.find('#00F5D4')],
    'Talent Market Intelligence': 'Talent Market Intelligence' in src,
    'Workforce Forecast Engine': 'Workforce Forecast Engine' in src,
    'Candidate Intelligence Center': 'Candidate Intelligence Center' in src,
    'Boardroom Intelligence': 'Boardroom Intelligence' in src,
    'Global Talent Radar': 'Global Talent Radar' in src,
}
for k, v in checks.items():
    print('{}: {}'.format(k, 'YES' if v else 'no'))

# Check sidebar
sidebar_start = src.find("pages_nav = [")
sidebar_end = src.find("]", sidebar_start) + 1
if sidebar_start >= 0:
    sidebar = src[sidebar_start:sidebar_end]
    print('\nSidebar pages:')
    for line in sidebar.split('\n'):
        line = line.strip()
        if '(' in line and ',' in line and 'None' not in line:
            # Extract the label (second quoted string)
            parts = line.split(',')
            if len(parts) >= 2:
                label = parts[-1].strip().strip('"').strip("'")
                print('  - {}'.format(label.encode('ascii', 'replace').decode('ascii')))
