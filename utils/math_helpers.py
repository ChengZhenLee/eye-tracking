import math
import numpy as np

class CoordinatesMath:
    def midpoint(p1: tuple[int], p2: tuple[int]) -> tuple[int]:
        midX = (p1[0] + p2[0]) // 2
        midY = (p1[1] + p2[1]) // 2
        return (midX, midY)


    def distance(p1: tuple[int], p2: tuple[int]) -> int:
        xDistSquared = (p1[0] - p2[0]) ** 2
        yDistSquared = (p1[1] - p2[1]) ** 2
        distance = math.sqrt(xDistSquared + yDistSquared)

        return distance


class NpMath:
    def getRegionCorner(coords: np.ndarray) -> np.ndarray:
        minX = np.min(coords[:, 0])
        maxX = np.max(coords[:, 0])
        minY = np.min(coords[:, 1])
        maxY = np.max(coords[:, 1])

        return np.array([minX, maxX, minY, maxY], np.int32)