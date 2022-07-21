import torch

# https://www.cnblogs.com/codehome/p/16453848.html

if __name__ == '__main__':
    content_1 = torch.load('cmt_tiny.pth')
    content_2 = torch.load('faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

    # 修改键值
    old_val = content_1['model']
    content_1['state_dict'] = content_1.pop('model')
    torch.save(content_1, './cmt_tiny_mm.pth')

    changed_dict = torch.load('./cmt_tiny_mm.pth')
    print(old_val.keys())
    print(changed_dict['state_dict'].keys())

    # 查看键值
    # print(content_1.keys())
    # print(content_2.keys())
    #
    # print(content_1['state_dict'].keys())
    # print(content_2['state_dict'].keys())
    # print(content['state_dict'].keys())
