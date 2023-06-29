import pickle
danger_label=[42,43,44,45,46,47,49,50,51,56,105,106,107,108,109]
f=open('ntu120_hrnet.pkl', 'rb')
out=open('danger_hrnet.pkl','wb')
data = pickle.load(f)
danger_val=[]
danger_train=[]
for video in data['annotations']:
    if(video['label'] in danger_label):
        frame_dir=video['frame_dir']
        if(frame_dir in data['split']['xsub_train']):
            danger_train.append(frame_dir)
        else:
            danger_val.append(frame_dir)
data['split']['danger_xsub_train']=danger_train
data['split']['danger_xsub_val']=danger_val
out.write(out)