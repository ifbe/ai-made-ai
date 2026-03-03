import torch

def print_torch_backends():
    """遍历并打印PyTorch后端信息"""
    print("PyTorch后端信息:")
    print("-" * 30)
    
    # 获取所有torch.backends下的属性
    backends = dir(torch.backends)
    
    for backend in backends:
        if not backend.startswith('_'):  # 过滤掉私有属性
            attr = getattr(torch.backends, backend)
            print(f"torch.backends.{backend}: {attr}")

if __name__ == "__main__":
    print_torch_backends()
