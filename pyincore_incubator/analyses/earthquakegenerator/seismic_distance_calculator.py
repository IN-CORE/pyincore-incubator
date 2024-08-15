# Copyright (c) 2024 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import numpy as np


class SeismicDistanceCalculator:
    @staticmethod
    def calculate_rupture_distance(site_loc, source_loc, strike, dip):
        """
        Calculates the shortest distance from a site location to a rupture plane (Rrup).

        Parameters:
        -----------
        site_loc : tuple of float
            Coordinates of site location (longitude, latitude, depth) in decimal degrees and km.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (longitude, latitude, depth) in decimal degrees and km.
        strike : float
            Strike angle of the rupture plane in degrees.
        dip : float
            Dip angle of the rupture plane in degrees.

        Returns:
        --------
        float: Shortest distance from the site location to the rupture plane in km (Rrup).
        """
        site_lon, site_lat, site_depth = site_loc
        source_lon, source_lat, source_depth = source_loc

        # Earth radius in km
        R = 6371

        # Convert latitude and longitude from degrees to radians
        site_lon_rad, site_lat_rad, source_lon_rad, source_lat_rad = map(
            np.radians, [site_lon, site_lat, source_lon, source_lat]
        )

        # Compute differences in coordinates
        dLon = source_lon_rad - site_lon_rad
        dLat = source_lat_rad - site_lat_rad

        # Haversine formula to calculate horizontal distance
        a = (
            np.sin(dLat / 2) ** 2
            + np.cos(site_lat_rad) * np.cos(source_lat_rad) * np.sin(dLon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        horizontal_distance = R * c

        # Calculate the vertical distance
        vertical_distance = abs(source_depth - site_depth)

        # Total rupture distance
        Rrup = np.sqrt(horizontal_distance**2 + vertical_distance**2)

        return Rrup

    @staticmethod
    def calculate_joyner_boore_distance(site_loc, source_loc):
        """
        Calculates the Joyner-Boore distance (Rjb) between two points based on their longitude and latitude.

        Parameters:
        -----------
        site_loc : tuple of float
            Coordinates of site location (longitude, latitude, depth) in decimal degrees.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (longitude, latitude, depth) in decimal degrees.

        Returns:
        --------
        float: Joyner-Boore distance between the site and source in km (Rjb).
        """
        # Extract latitude and longitude from the input tuples
        site_lon, site_lat, _ = site_loc
        source_lon, source_lat, _ = source_loc

        # Radius of the Earth in kilometers
        R = 6371

        # Convert latitude and longitude from degrees to radians
        site_lon, site_lat, source_lon, source_lat = map(
            np.radians, [site_lon, site_lat, source_lon, source_lat]
        )

        # Haversine formula
        dlon = source_lon - site_lon
        dlat = source_lat - site_lat
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(site_lat) * np.cos(source_lat) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Distance in kilometers
        Rjb = R * c

        return Rjb

    @staticmethod
    def calculate_horizontal_distance(site_loc, source_loc, strike):
        """
        Calculates the horizontal distance (Rx) from the top edge of the rupture to a site location,
        measured perpendicular to the fault strike.

        Parameters:
        -----------
        site_loc : tuple of float
            Coordinates of site location (longitude, latitude, depth) in decimal degrees and km.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (longitude, latitude, depth) in decimal degrees and km.
        strike : float
            Fault strike angle in degrees.

        Returns:
        --------
        float: Horizontal distance from the top edge of the rupture to the site location, measured perpendicular
              to the fault strike in km (Rx).
        """
        # Convert strike to radians
        strike_rad = np.radians(strike)

        # Extract coordinates
        site_lon, site_lat, site_depth = site_loc
        source_lon, source_lat, source_depth = source_loc

        # Convert latitude and longitude from degrees to radians for calculation
        site_lon_rad, site_lat_rad, source_lon_rad, source_lat_rad = map(
            np.radians, [site_lon, site_lat, source_lon, source_lat]
        )

        # Calculate the differences
        dlon = site_lon_rad - source_lon_rad
        dlat = site_lat_rad - source_lat_rad

        # Approximate distance in the east direction (longitude) and north direction (latitude)
        R = 6371  # Radius of the Earth in km
        delta_x = (
            R * dlon * np.cos((site_lat_rad + source_lat_rad) / 2)
        )  # East direction
        delta_y = R * dlat  # North direction
        delta_z = site_depth - source_depth  # Vertical direction

        # Create vectors
        strike_vector = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0])
        top_edge_to_site = np.array([delta_x, delta_y, delta_z])

        # Calculate distance from site to closest point on rupture trace
        dist_to_rupture_trace = np.dot(top_edge_to_site, strike_vector)
        Rx = abs(dist_to_rupture_trace)

        return Rx

    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the Haversine distance between two points based on their latitude and longitude coordinates.

        Parameters:
        -----------
        lat1, lon1 : float
            Latitude and longitude of the first point in degrees.
        lat2, lon2 : float
            Latitude and longitude of the second point in degrees.

        Returns:
        --------
        float: The Haversine distance between the two points in km (Rh).
        """
        R = 6371  # Earth radius in kilometers

        # Convert latitude and longitude from degrees to radians
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
            np.radians, [lon1, lat1, lon2, lat2]
        )

        # Compute differences in coordinates
        dLon = lon2_rad - lon1_rad
        dLat = lat2_rad - lat1_rad

        # Haversine formula
        a = (
            np.sin(dLat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dLon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        Rh = R * c

        return Rh
