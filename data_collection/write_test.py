from get_address_info import GetAddressInfo

new_address = GetAddressInfo("bc1qdm7xde6lcauqx2zy5mf2fe934pfk0yyllgehlg",0)
new_address.address_write()
a = new_address.a
print(a)