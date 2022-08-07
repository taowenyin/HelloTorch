import torch

# https://www.cnblogs.com/codehome/p/16453848.html
# https://www.cnblogs.com/sddai/p/14949982.html


class PthUtil:
    def __init__(self):
        print('===初始化PTH工具===')

    def view_keys(self, pth_model):
        return pth_model.keys()

    def save_model(self, pth_model, save_path):
        torch.save(pth_model, save_path)

    def modify_key_name(self, pth_path, root_key, old_key, new_key):
        pth_model = torch.load(pth_path, map_location=torch.device('cpu'))
        pth_model[root_key][new_key] = pth_model[root_key].pop(old_key)

        torch.save(pth_model, pth_path)

    def modify_root_key_name(self, pth_path, old_key, new_key):
        pth_model = torch.load(pth_path, map_location=torch.device('cpu'))
        pth_model[new_key] = pth_model.pop(old_key)

        torch.save(pth_model, pth_path)

    def delete_keys(self, pth_path, root_key, keys):
        pth_model = torch.load(pth_path, map_location=torch.device('cpu'))
        # key.startswith('decoder1')
        for key in list(pth_model[root_key].keys()):
            if key in keys:
                del pth_model[root_key][key]

        torch.save(pth_model, pth_path)

    def change_key_value(self, pth_path, root_key, key, value):
        pth_model = torch.load(pth_path, map_location=torch.device('cpu'))
        for _key in list(pth_model[root_key].keys()):
            if _key == key:
                pth_model[root_key][_key] = value
                break

        torch.save(pth_model, pth_path)


if __name__ == '__main__':
    util = PthUtil()

    util.modify_root_key_name('./cmt_xs.pth', 'model', 'state_dict')

    pth_model = torch.load('./cmt_xs.pth', map_location=torch.device('cpu'))

    print(util.view_keys(pth_model))

    # util.delete_keys('./cmt_tiny_mm.pth', 'state_dict', ['head.weight', 'head.bias', '_fc.weight', '_fc.bias',
    #                                                      '_bn.weight', '_bn.bias', '_bn.running_mean',
    #                                                      '_bn.running_var', '_bn.num_batches_tracked',
    #                                                      'relative_pos_a', 'relative_pos_b',
    #                                                      'relative_pos_c', 'relative_pos_d'])


    # # 修改键值
    # old_val = content_1['model']
    # content_1['state_dict'] = content_1.pop('model')
    # torch.save(content_1, './cmt_tiny_mm.pth')
    #
    # changed_dict = torch.load('./cmt_tiny_mm.pth')
    # print(old_val.keys())
    # print(changed_dict['state_dict'].keys())

    # 查看键值
    # print(content_1.keys())
    # print(content_2.keys())
    #
    # print(content_1['state_dict'].keys())
    # print(content_2['state_dict'].keys())
    # print(content['state_dict'].keys())
