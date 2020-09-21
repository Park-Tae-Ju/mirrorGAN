import pymysql
import json
import os
import os.path as osp
from glob import glob

if __name__ == "__main__":
    with open('database_properties.json') as f:
        db_info = json.loads(f.read())
    conn = pymysql.connect(**db_info)

    count = 0

    # read files
    ROOT_DIR = osp.join(os.getcwd(), '..')
    DATA_DIR = osp.join(ROOT_DIR, 'data', 'birds', 'text')
    for folder in glob(r'{}/*'.format(DATA_DIR)):
        if osp.isdir(folder):
            folder_no = osp.basename(folder).split('.')[0]
            count += 1
            for file in glob(r'{}/*.txt'.format(folder)):
                insert_sql = """insert into a_text_bird (folder_no, file_name, caption) values (%s, %s, %s)"""
                with open(file) as f:
                    texts = f.readlines()
                data = []
                for t in texts:
                    data.append((folder_no, osp.basename(file).replace('txt', 'jpg'), t))

                with conn.cursor() as cursor:
                    cursor.executemany(insert_sql, data)
                    conn.commit()

    print('folder: %d' % count)

