from json import load

filename = 'Football-Outcome-Prediction.ipynb'
with open("output.py", "w") as o_file:
    with open(filename) as fp:
        nb = load(fp)

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(line for line in cell['source'] if not line.startswith('%'))
                o_file.write(source)
                o_file.write("\n")
                # exec(source, globals(), locals())


