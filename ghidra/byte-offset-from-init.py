# Ghidrathon script, python 3
#@menupath Tools.Copy Byte Offset from init

# NOTE: Supports macOS only

import utils
import subprocess

init_block = list(filter(lambda x: x.getName() == '.init', utils.currentProgram().getMemory().getBlocks()))[0]
assert init_block

init_addr = init_block.getStart().getOffset()
curr_addr = utils.currentLocation().getByteAddress().getOffset()

subprocess.run('pbcopy', universal_newlines=True, input=f'{curr_addr-init_addr:x}')
