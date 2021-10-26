def write_cfg(dir, name, op, **kwargs):


    f = open(f"{dir}/{name}.txt", op)
    if op == 'w':
        f.write(f"dir:{dir}\nname: {name}\n")
    for key,value in kwargs.items():
        f.write(f"{key}: {value}\n")

    f.close()