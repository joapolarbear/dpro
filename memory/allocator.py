

class Block:
    def __init__(self, size, pool):
        self.size = size
        self.pool = pool
        self.allocated = False
        self.prev = None
        self.next = None

    def is_split(self):
        return (self.prev is not None) or \
            (self.next is not None)

    def __repr__(self):
        return "block info: size=%d allocated=%d has_prev=%d " \
            "has_next=%d is_split = %d" % (
                self.size, self.allocated is True,
                self.prev is not None,
                self.next is not None, self.is_split()
            )


class BlockPool:
    def __init__(self):
        self.pool = set()

    def insert(self, block):
        self.pool.add(block)

    def remove(self, block):
        if block in self.pool:
            self.pool.remove(block)
        else:
            print(block)
            for _, blk in self.pool:
                print(blk)
            raise ValueError("remove: not found")

    def find(self, search_key):
        s = sorted(self.pool, key=lambda x: x.size)
        for blk in s:
            if blk.size >= search_key.size:
                self.pool.remove(blk)
                return blk
        return None

    def __len__(self):
        return len(self.pool)


class AllocParams:
    def __init__(self, size, pool, alloc_size):
        self.search_key = Block(size, pool)
        self.pool = pool
        self.alloc_size = alloc_size
        self.block = None


kMinBlockSize = 512
kSmallSize = 1048576  # 1MB
kSmallBuffer = 2097152  # 2MB
kLargeBuffer = 20971520  # 20MB
kMinLargeAlloc = 10485760  # 10MB
kRoundLarge = 2097152  # 2MB


def round_size(size):
    if size < kMinBlockSize:
        return kMinBlockSize
    else:
        return kMinBlockSize*((size + kMinBlockSize - 1) // kMinBlockSize)


def get_allocation_size(size):
    if size <= kSmallSize:
        return kSmallBuffer
    elif size < kMinLargeAlloc:
        return kLargeBuffer
    else:
        return kRoundLarge * ((size + kRoundLarge - 1) // kRoundLarge)


class Stat:
    def __init__(self, init=0):
        self.current = init
        self.peak = 0

    def update(self, amount):
        self.current += amount
        self.peak = max(self.current, self.peak)


class CachingAllocator:

    def __init__(self, budget, cuda_context=0):
        self.large_blocks = BlockPool()  # unallocated
        self.small_blocks = BlockPool()  # unallocated
        self.active_blocks = BlockPool()
        self.budget = budget
        self.reserved = Stat(cuda_context)  # CUDA Context
        self.activated = Stat()
        self.inactive = Stat()

    def get_pool(self, size):
        if size <= kSmallSize:
            return self.small_blocks
        return self.large_blocks

    def get_free_block(self, params):
        blk = params.pool.find(params.search_key)
        if blk:
            params.block = blk
            return True
        return False

    def alloc_block(self, params):
        size = params.alloc_size
        # TODO: cudaMalloc error. fragmentation
        if self.reserved.current + size >= self.budget:
            return False
        params.block = Block(size, params.pool)
        # cudaMalloc
        self.reserved.update(size)
        return True

    def free_blocks(self, pool):
        """free all non-split blocks
        """
        for blk in list(pool.pool):
            if blk.prev is None and blk.next is None:
                # cudaFree
                self.reserved.update(-blk.size)
                pool.remove(blk)

        return True

    def free_cached_blocks(self):
        self.free_blocks(self.large_blocks)
        self.free_blocks(self.small_blocks)
        return True

    def should_split(self, block, size):
        remaining = block.size - size
        if block.pool is self.small_blocks:
            return remaining >= kMinBlockSize
        elif block.pool is self.large_blocks:
            return remaining > kSmallSize
        else:
            raise ValueError("should_split: invalid pool")

    def try_merge_blocks(self, dst, src, pool):
        if src is None or src.allocated:
            return 0

        assert dst.is_split() and src.is_split()

        if dst.prev is src:
            dst.prev = src.prev
            if dst.prev:
                dst.prev.next = dst
        else:
            dst.next = src.next
            if dst.next:
                dst.next.prev = dst

        subsumed_size = src.size
        dst.size += subsumed_size
        pool.remove(src)
        return subsumed_size

    def free_block(self, block):
        pool = block.pool
        merge_candidates = [block.prev, block.next]
        net_change_inactive_split_size = 0
        for merge_candidate in merge_candidates:
            subsumed_size = self.try_merge_blocks(block, merge_candidate, pool)
            if subsumed_size > 0:
                net_change_inactive_split_size -= subsumed_size

        self.activated.update(-block.size)
        self.active_blocks.remove(block)
        pool.insert(block)

        if block.is_split():
            net_change_inactive_split_size += block.size

        self.inactive.update(net_change_inactive_split_size)

    def malloc(self, size):
        if size == 0:
            return None

        size = round_size(size)
        pool = self.get_pool(size)
        alloc_size = get_allocation_size(size)
        params = AllocParams(size, pool, alloc_size)

        block_find = self.get_free_block(params) or \
            self.alloc_block(params) or \
            (self.free_cached_blocks() and self.alloc_block(params))

        if not block_find:
            raise RuntimeError("OOM")

        block = params.block
        already_split = block.is_split()
        if self.should_split(block, size):
            remaining = block

            block = Block(size, pool)
            block.prev = remaining.prev
            if block.prev:
                block.prev.next = block
            block.next = remaining

            remaining.prev = block
            remaining.size -= size
            pool.insert(remaining)

            if already_split:
                self.inactive.update(-block.size)
            else:
                self.inactive.update(remaining.size)
        elif already_split:
            self.inactive.update(-block.size)

        assert block.allocated is False
        block.allocated = True
        self.activated.update(block.size)
        self.active_blocks.insert(block)

        return block

    def free(self, block):
        block.allocated = False
        self.free_block(block)

    def empty_cache(self):
        self.free_cached_blocks()

    def get_max_block_size_aux(self, pool, largest):
        for blk in pool.pool:
            if blk.size > largest:
                largest = blk.size
        return largest

    def get_max_block_size(self):
        # optimistic value
        largest = self.budget - self.reserved.current
        largest = self.get_max_block_size_aux(self.large_blocks, largest)
        largest = self.get_max_block_size_aux(self.small_blocks, largest)
        return largest

    def show_all_blocks(self, pool):
        print("blocks num=%d" % (len(pool)))
        for blk in pool.pool:
            print(blk)

    def debug(self):
        print("active blocks")
        self.show_all_blocks(self.active_blocks)

        print("small blocks")
        self.show_all_blocks(self.small_blocks)

        print("large blocks")
        self.show_all_blocks(self.large_blocks)

        print("peak memory=%.2f" % (self.reserved.peak / 1024.0 / 1024.0))
