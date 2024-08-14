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
            Coordinates of site location (latitude, longitude, depth) in decimal degrees and km.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (latitude, longitude, depth) in decimal degrees and km.
        strike : float
            Strike angle of the rupture plane in degrees.
        dip : float
            Dip angle of the rupture plane in degrees.

        Returns:
        --------
        float: Shortest distance from the site location to the rupture plane in km (Rrup).
        """
        x1, y1, z1 = site_loc
        x2, y2, z2 = source_loc

        strike_radians = np.radians(strike)
        dip_radians = np.radians(dip)

        # Calculate normal vector to rupture plane
        nx = np.sin(dip_radians) * np.sin(strike_radians)
        ny = -np.sin(dip_radians) * np.cos(strike_radians)
        nz = np.cos(dip_radians)

        # Calculate distance from site location to rupture plane
        Rrup = abs(nx * (x1 - x2) + ny * (y1 - y2) + nz * (z1 - z2)) / np.sqrt(
            nx**2 + ny**2 + nz**2
        )

        return Rrup

    @staticmethod
    def calculate_joyner_boore_distance(site_loc, source_loc):
        """
        Calculates the Joyner-Boore distance (Rjb) between two points based on their longitude and latitude.

        Parameters:
        -----------
        site_loc : tuple of float
            Coordinates of site location (latitude, longitude) in decimal degrees.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (latitude, longitude) in decimal degrees.

        Returns:
        --------
        float: Joyner-Boore distance between the site and source in km (Rjb).
        """
        # Extract latitude and longitude from the input tuples
        site_lat, site_lon, depth = site_loc
        source_lat, source_lon, depth = source_loc

        # Radius of the Earth in kilometers
        R = 6371

        # Convert latitude and longitude from degrees to radians
        site_lat, site_lon, source_lat, source_lon = map(
            np.radians, [site_lat, site_lon, source_lat, source_lon]
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
            Coordinates of site location (latitude, longitude, depth) in decimal degrees and km.
        source_loc : tuple of float
            Coordinates of the top edge of the rupture plane (latitude, longitude, depth) in decimal degrees and km.
        strike : float
            Fault strike angle in degrees.

        Returns:
        --------
        float: Horizontal distance from the top edge of the rupture to the site location, measured perpendicular
               to the fault strike in km (Rx).
        """
        # Convert strike to radians
        strike_rad = np.radians(strike)

        # Calculate unit vector along strike direction
        strike_vector = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0])

        # Calculate vector from top edge of rupture plane to site
        top_edge_to_site = np.array(site_loc) - np.array(source_loc)

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

        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)

        a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        Rh = R * c

        return Rh
