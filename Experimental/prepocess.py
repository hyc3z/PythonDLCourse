import os
val_dir1='/home/hu/Downloads/ILSVRC/Data/CLS-LOC/val'
val_image_list_file = open('/home/hu/Downloads/LOC_val_solution.csv')
val_image_labels = {}
val_dir = []
last_dir = ''
while val_image_list_file:
    a = val_image_list_file.readline()
    try:
        q = a.split(',')
        if len(q) != 0 :
            val_image_labels[q[0]] = q[1].split(' ')[0]
            if not q[1].split(' ')[0] in val_dir:
                val_dir.append(q[1].split(' ')[0])
    except IndexError:
            break

for i in val_dir:
    print('mkdir -p '+os.path.join(val_dir1, i))
    os.system('mkdir -p '+os.path.join(val_dir1, i))

for i in val_image_labels:
    print('mv '+os.path.join(val_dir1, i)+'.JPEG '+os.path.join(val_dir1, val_image_labels[i]))
    os.system('mv '+os.path.join(val_dir1, i)+'.JPEG /'+os.path.join(val_dir1, val_image_labels[i]))
