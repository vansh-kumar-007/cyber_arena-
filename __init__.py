$inits = @(
    "env\__init__.py",
    "agents\__init__.py",
    "configs\__init__.py",
    "utils\__init__.py",
    "visualization\__init__.py"
)
foreach ($f in $inits) { New-Item -ItemType File -Path $f -Force }