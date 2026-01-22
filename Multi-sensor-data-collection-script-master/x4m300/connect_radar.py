import pymoduleconnector
import time

device_name = "COM11"

mc = pymoduleconnector.ModuleConnector(device_name)
xep = mc.get_xep()

# 检查模块是否已经连接, “0xaaeeaeea” means system ready and “0xaeeaeeaa” means system not ready
print("receive ping, the value is:", hex(xep.ping()))

# 查看xep中有哪些方法可以用
print(dir(xep))

#重置模块，模块将重启。
xep.module_reset()
mc.close()
time.sleep(3)