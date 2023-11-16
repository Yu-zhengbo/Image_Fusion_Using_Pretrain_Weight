import xml.etree.ElementTree as ET
import os 
def read_xml_file(file_path):
    # 加载并解析xml文件
    tree = ET.parse(file_path)

    # 获取根节点
    root = tree.getroot()
    names = []
    # 查找所有的object元素
    for obj in root.findall('object'):
        # 获取name元素的文本
        name = obj.find('name').text
        names.append(name)
        # 获取bndbox元素的信息
        # bndbox = obj.find('bndbox')
        # xmin = bndbox.find('xmin').text
        # ymin = bndbox.find('ymin').text
        # xmax = bndbox.find('xmax').text
        # ymax = bndbox.find('ymax').text

    return names


#获取数据集有哪些检测的目标
if __name__ == "__main__":
    base_path = '/data/models/patent/datasets/M3FD_with_det/ann'
    xml_list = os.listdir(base_path)

    cls_all = []
    for xml in xml_list:
        xml = os.path.join(base_path,xml)
        cls_temp  = read_xml_file(xml)
        cls_all.extend(cls_temp)
    print(list(set(cls_all)))

    '''
    ['Car', 'People', 'Motorcycle', 'Truck', 'Bus', 'Lamp']
    '''