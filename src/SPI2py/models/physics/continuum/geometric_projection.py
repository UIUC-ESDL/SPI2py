# import math
#
# def circle_overlap_area(r1, r2, d):
#     """
#     Calculate the area of overlap between two circles with radii r1 and r2,
#     and distance d between their centers.
#
#     :param r1: Radius of the first circle
#     :param r2: Radius of the second circle
#     :param d: Distance between the centers of the two circles
#     :return: Overlapping area
#     """
#
#     # If one circle is completely within the other
#     if d <= abs(r1 - r2):
#         return math.pi * min(r1, r2) ** 2
#
#     # If the circles do not overlap
#     if d >= r1 + r2:
#         return 0
#
#     # Calculate overlap area
#     r1_sq, r2_sq = r1 ** 2, r2 ** 2
#     alpha = math.acos((r1_sq + d**2 - r2_sq) / (2 * r1 * d))
#     beta = math.acos((r2_sq + d**2 - r1_sq) / (2 * r2 * d))
#     area1 = r1_sq * alpha - r1_sq * math.sin(2 * alpha) / 2
#     area2 = r2_sq * beta - r2_sq * math.sin(2 * beta) / 2
#
#     return area1 + area2
#
# # Example usage
# circle_overlap_area(3, 4, 5)  # Example radii and distance between centers
#
#
# def distance_between_points(p1, p2):
#     """
#     Calculate the Euclidean distance between two points.
#
#     :param p1: Coordinates of the first point (x1, y1)
#     :param p2: Coordinates of the second point (x2, y2)
#     :return: Distance between the two points
#     """
#     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#
# def overlap_percentage(set1, set2):
#     """
#     Calculate the percentage of area overlap for each circle in set1 with all circles in set2.
#
#     :param set1: List of circles in set 1, where each circle is represented as (radius, (x, y))
#     :param set2: List of circles in set 2, where each circle is represented as (radius, (x, y))
#     :return: List of overlap percentages for each circle in set1
#     """
#     overlap_percentages = []
#
#     for circle1 in set1:
#         total_overlap_area = 0
#         for circle2 in set2:
#             d = distance_between_points(circle1[1], circle2[1])
#             overlap_area = circle_overlap_area(circle1[0], circle2[0], d)
#             total_overlap_area += overlap_area
#
#         circle1_area = math.pi * circle1[0] ** 2
#         overlap_percent = (total_overlap_area / circle1_area) * 100 if circle1_area > 0 else 0
#         overlap_percentages.append(overlap_percent)
#
#     return overlap_percentages
#
# # Example usage
# set1 = [(3, (0, 0)), (2, (5, 5))]  # Two circles in set 1 with radii 3 and 2 and centers at (0,0) and (5,5)
# set2 = [(4, (3, 3)), (1, (6, 6))]  # Two circles in set 2 with radii 4 and 1 and centers at (3,3) and (6,6)
#
# overlap_percentage(set1, set2)  # Calculate overlap percentages for each circle in set1 with circles in set2
#
#
# import numpy as np
#
# def vectorized_overlap_percentage(set1, set2):
#     """
#     Vectorized calculation of the percentage of area overlap for each circle in set1 with all circles in set2.
#
#     :param set1: List of circles in set 1, where each circle is represented as (radius, (x, y))
#     :param set2: List of circles in set 2, where each circle is represented as (radius, (x, y))
#     :return: Array of overlap percentages for each circle in set1
#     """
#     # Extract radii and center coordinates for each set
#     radii1, centers1 = zip(*set1)
#     radii2, centers2 = zip(*set2)
#
#     # Convert lists to numpy arrays
#     radii1 = np.array(radii1)
#     centers1 = np.array(centers1)
#     radii2 = np.array(radii2)
#     centers2 = np.array(centers2)
#
#     # Calculate pairwise distances between centers of circles in set1 and set2
#     dist_matrix = np.sqrt(np.sum((centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]) ** 2, axis=2))
#
#     # Calculate pairwise overlap areas
#     r1, r2 = np.meshgrid(radii1, radii2)
#     d = dist_matrix
#
#     # Handle complete overlap and no overlap cases
#     complete_overlap = np.where(d <= np.abs(r1 - r2), np.pi * np.minimum(r1, r2) ** 2, 0)
#     no_overlap = np.where(d >= r1 + r2, 0, complete_overlap)
#
#     # Calculate overlap area using the formula
#     term1 = r1 ** 2 * np.arccos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
#     term2 = r2 ** 2 * np.arccos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
#     term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
#     overlap_areas = term1 + term2 - term3
#
#     # Replace complete overlap and no overlap cases
#     overlap_areas = np.where(d <= np.abs(r1 - r2), complete_overlap, overlap_areas)
#     overlap_areas = np.where(d >= r1 + r2, no_overlap, overlap_areas)
#
#     # Calculate total overlap area for each circle in set1
#     total_overlap_areas = np.sum(overlap_areas, axis=1)
#
#     # Compute the percentage of overlap
#     circle1_areas = np.pi * radii1 ** 2
#     overlap_percentages = (total_overlap_areas / circle1_areas) * 100
#
#     return overlap_percentages
#
# # Example usage with the same sets of circles
# vectorized_overlap_percentage(set1, set2)
