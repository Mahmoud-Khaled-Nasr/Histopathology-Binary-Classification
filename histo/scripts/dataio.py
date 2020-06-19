### Libraries
import glob

### Data I/O
def findScan(data, name, key):
    """"
    Input: 
        data     - dict of dict to sort the data
        key      - key of 'data' ('id','image','label')
        value    - value of 'key'
    Output:
        value
    """
    for i, dic in data.items():
        if dic[key] == name:        
            return i
    return -1




def sortData(path, mode='train-val'):
    """"
    Input:  path
    Output: data[p] = {
        'id':
        'image':
        'label': }
        
    """
    if (mode=='train-val'):
        # Importing Images
        B0_dir  = glob.glob(path+"/b0/*.png")
        M0_dir  = glob.glob(path+"/m0/*.png")

        print("Number of B0 Images:",  len(B0_dir))
        print("Number of M0 Images:",  len(M0_dir))

        # Creating Dictionary (B0 Scans)
        data = {}
        for p in range(len(B0_dir)):
            scan_id  =  B0_dir[p].replace(".png", "")
            scan_id  =  scan_id.replace(path+"/b0\\", "")

            # Creating list of dictionary                    
            data[p] = {
                        'id'    : scan_id,
                        'image' : B0_dir[p],
                        'label' : 0 }            # Label of B0 = 0

        # Creating Dictionary (M0 Scans)
        for p in range(len(M0_dir)):
            scan_id  =  M0_dir[p].replace(".png", "")
            scan_id  =  scan_id.replace(path+"/m0\\", "")

            # Creating list of dictionary                    
            data[p+len(M0_dir)] = {
                        'id'    : scan_id,
                        'image' : M0_dir[p],
                        'label' : 1 }            # Label of M0 = 1

    
    elif (mode=='test'):     
        # Importing Images
        target_dir  = glob.glob(path+"/*.png")
        print("Number of Test Images:", len(target_dir))
        
        # Creating Dictionary
        data = {}
        for p in range(len(target_dir)):
            scan_id  =  target_dir[p].replace(".jpg", "")
            scan_id  =  scan_id.replace(path+"\\", "")

            # Creating List of Dictionary                    
            data[p] = {
                        'id'    : scan_id,
                        'image' : target_dir[p] }         
    return data
