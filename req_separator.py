conda_packages = []
pip_packages = []

with open('req.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    if '=pypi_0' in line or '==0.0.0' in line:
        pip_packages.append(line)
    else:
        conda_packages.append(line)

with open('conda-requirements.txt', 'w') as f:
    for pkg in conda_packages:
        f.write(pkg)

with open('pip-requirements.txt', 'w') as f:
    for pkg in pip_packages:
        f.write(pkg)
