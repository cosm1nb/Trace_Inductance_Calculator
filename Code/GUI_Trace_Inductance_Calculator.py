#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:24:31 2026

@author: Cosmin Bondreanu

MIT-Licence
Copyright (c) 2026 Cosmin Bondreanu


Trace_Inductance_Calculator
Version: 1
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

import numpy as np
import matplotlib.colors as mcolors

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# DEFAULT VALUES
# ============================================================

DEFAULTS = {
    # Length entered by user in cm
    "l_min": 0.1,
    "l_max": 100.0,
    "l_samples": 100,

    # Width entered by user in mm
    "w_min": 0.1,
    "w_max": 20.0,
    "w_samples": 100,

    # Copper thickness entered by user in mm
    "t_mm": 0.035,

    # 2D fixed points
    "fixed_width": 5.0,       # mm
    "fixed_length": 1.0,      # cm

    # Default colormap
    "cmap": "inferno",
}


# ============================================================
# CALCULATION
# ============================================================

def calculate_inductance(
    l_min_cm,
    l_max_cm,
    l_samples,
    w_min_mm,
    w_max_mm,
    w_samples,
    t_mm
):
    """
    Terman Formula.

    User-facing units:
        Length    = cm
        Width     = mm
        Thickness = mm

    Internal calculation units:
        Length    = cm
        Width     = cm
        Thickness = cm

    Validity condition:
        L > 3W

    Everything inside this function uses cm after the
    initial conversion of width and thickness.
    """

    # --------------------------------------------------------
    # Convert user input from mm -> cm
    # --------------------------------------------------------

    w_min_cm = w_min_mm / 10.0
    w_max_cm = w_max_mm / 10.0

    t_cm = t_mm / 10.0

    # --------------------------------------------------------
    # Create 1D arrays
    # --------------------------------------------------------

    length_cm_1d = np.linspace(
        l_min_cm,
        l_max_cm,
        l_samples
    )

    width_cm_1d = np.linspace(
        w_min_cm,
        w_max_cm,
        w_samples
    )

    # --------------------------------------------------------
    # Create 2D grid
    # --------------------------------------------------------

    length_cm, width_cm = np.meshgrid(
        length_cm_1d,
        width_cm_1d
    )

    # --------------------------------------------------------
    # Terman Formula
    # --------------------------------------------------------

    L = (
        2.0
        * length_cm
        * (
            np.log(
                (2.0 * length_cm)
                / (width_cm + t_cm)
            )
            + 0.5
            + 0.2235
            * (
                (width_cm + t_cm)
                / length_cm
            )
        )
    )

    # ========================================================
    # VALIDITY MASK
    # ========================================================
    #
    # The Terman Formula is valid for:
    #
    #     L > 3W
    #
    # ========================================================

    valid = length_cm > (3.0 * width_cm)

    L_masked = np.ma.masked_where(
        ~valid,
        L
    )

    return (
        length_cm,
        width_cm,
        L_masked
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class PCBTraceInductanceApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "PCB Trace Inductance Calculator"
        )

        self.root.geometry(
            "1400x850"
        )

        self.root.minsize(
            1100,
            700
        )

        # ----------------------------------------------------
        # Calculation data
        # ----------------------------------------------------

        self.length_grid = None
        self.width_grid = None
        self.inductance = None

        # ----------------------------------------------------
        # Matplotlib objects
        # ----------------------------------------------------

        self.surface = None
        self.colorbar = None

        # ----------------------------------------------------
        # Tk variables
        # ----------------------------------------------------

        self.l_min_var = tk.StringVar()
        self.l_max_var = tk.StringVar()
        self.l_samples_var = tk.StringVar()

        self.w_min_var = tk.StringVar()
        self.w_max_var = tk.StringVar()
        self.w_samples_var = tk.StringVar()

        self.t_var = tk.StringVar()

        self.cmap_var = tk.StringVar()

        self.two_d_variable_var = tk.StringVar(
            value="Length"
        )

        self.fixed_point_var = tk.StringVar()

        self.status_var = tk.StringVar(
            value="Ready"
        )

        # ----------------------------------------------------
        # Build interface
        # ----------------------------------------------------

        self.create_menu()
        self.create_main_layout()

        self.reset_defaults()
        self.calculate()

    # ========================================================
    # MENU
    # ========================================================

    def create_menu(self):

        menu = tk.Menu(
            self.root
        )

        file_menu = tk.Menu(
            menu,
            tearoff=0
        )

        file_menu.add_command(
            label="Save Graph...",
            command=self.save_graph
        )

        file_menu.add_separator()

        file_menu.add_command(
            label="Exit",
            command=self.root.destroy
        )

        menu.add_cascade(
            label="File",
            menu=file_menu
        )

        self.root.config(
            menu=menu
        )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def create_main_layout(self):

        # ----------------------------------------------------
        # Input panel
        # ----------------------------------------------------

        self.input_frame = ttk.Frame(
            self.root,
            padding=15
        )

        self.input_frame.pack(
            side=tk.LEFT,
            fill=tk.Y
        )

        # ----------------------------------------------------
        # Graph area
        # ----------------------------------------------------

        self.graph_frame = ttk.Frame(
            self.root,
            padding=5
        )

        self.graph_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ttk.Label(
            self.input_frame,
            text="PCB Trace Inductance",
            font=("Arial", 18, "bold")
        ).pack(
            pady=(0, 5)
        )

        ttk.Label(
            self.input_frame,
            text="Terman Formula",
            font=("Arial", 11)
        ).pack(
            pady=(0, 20)
        )

        # ----------------------------------------------------
        # Trace geometry
        # ----------------------------------------------------

        geometry_frame = ttk.LabelFrame(
            self.input_frame,
            text="Trace Geometry",
            padding=10
        )

        geometry_frame.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        self.create_input(
            geometry_frame,
            "Length minimum",
            self.l_min_var,
            "cm"
        )

        self.create_input(
            geometry_frame,
            "Length maximum",
            self.l_max_var,
            "cm"
        )

        self.create_input(
            geometry_frame,
            "Length samples",
            self.l_samples_var,
            ""
        )

        ttk.Separator(
            geometry_frame,
            orient=tk.HORIZONTAL
        ).pack(
            fill=tk.X,
            pady=8
        )

        self.create_input(
            geometry_frame,
            "Width minimum",
            self.w_min_var,
            "mm"
        )

        self.create_input(
            geometry_frame,
            "Width maximum",
            self.w_max_var,
            "mm"
        )

        self.create_input(
            geometry_frame,
            "Width samples",
            self.w_samples_var,
            ""
        )

        # ----------------------------------------------------
        # Copper
        # ----------------------------------------------------

        copper_frame = ttk.LabelFrame(
            self.input_frame,
            text="Copper",
            padding=10
        )

        copper_frame.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        self.create_input(
            copper_frame,
            "Thickness",
            self.t_var,
            "mm"
        )

        ttk.Label(
            copper_frame,
            text="1 oz copper ≈ 0.035 mm",
            foreground="gray"
        ).pack(
            anchor="w",
            pady=(5, 0)
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_frame = ttk.LabelFrame(
            self.input_frame,
            text="Display",
            padding=10
        )

        display_frame.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        ttk.Label(
            display_frame,
            text="Colormap"
        ).pack(
            anchor="w"
        )

        colormaps = [
            "inferno",
            "viridis",
            "plasma",
            "magma",
            "cividis",
            "turbo",
            "twilight",
            "coolwarm",
            "RdYlBu",
            "Spectral"
        ]

        self.cmap_combo = ttk.Combobox(
            display_frame,
            textvariable=self.cmap_var,
            values=colormaps,
            state="readonly"
        )

        self.cmap_combo.pack(
            fill=tk.X,
            pady=(5, 0)
        )

        self.cmap_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.calculate()
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = ttk.Frame(
            self.input_frame
        )

        button_frame.pack(
            fill=tk.X,
            pady=10
        )

        ttk.Button(
            button_frame,
            text="Calculate",
            command=self.calculate
        ).pack(
            fill=tk.X,
            pady=3
        )

        ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset_defaults
        ).pack(
            fill=tk.X,
            pady=3
        )

        ttk.Button(
            button_frame,
            text="Save Graph...",
            command=self.save_graph
        ).pack(
            fill=tk.X,
            pady=3
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        ttk.Label(
            self.input_frame,
            textvariable=self.status_var,
            foreground="gray",
            wraplength=250
        ).pack(
            pady=10
        )

        # ----------------------------------------------------
        # Formula validity
        # ----------------------------------------------------

        validity_frame = ttk.LabelFrame(
            self.input_frame,
            text="Formula validity",
            padding=8
        )

        validity_frame.pack(
            fill=tk.X,
            pady=(5, 0)
        )

        ttk.Label(
            validity_frame,
            text="Length > 3 × Width",
            foreground="#555555"
        ).pack(
            anchor="w"
        )

        ttk.Label(
            validity_frame,
            text=(
                "The calculation is masked wherever "
                "this condition is not satisfied."
            ),
            foreground="#777777",
            wraplength=230
        ).pack(
            anchor="w",
            pady=(3, 0)
        )

        # ----------------------------------------------------
        # Notebook
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            self.graph_frame
        )

        self.notebook.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.tab_3d = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.tab_3d,
            text="3D Surface"
        )

        self.tab_2d = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.tab_2d,
            text="2D Plot"
        )

        self.create_3d_plot()
        self.create_2d_plot()

    # ========================================================
    # INPUT WIDGET
    # ========================================================

    def create_input(
        self,
        parent,
        label,
        variable,
        unit
    ):

        frame = ttk.Frame(
            parent
        )

        frame.pack(
            fill=tk.X,
            pady=3
        )

        ttk.Label(
            frame,
            text=label
        ).pack(
            side=tk.LEFT
        )

        entry = ttk.Entry(
            frame,
            textvariable=variable,
            width=12
        )

        entry.pack(
            side=tk.RIGHT,
            padx=5
        )

        if unit:

            ttk.Label(
                frame,
                text=f"[{unit}]"
            ).pack(
                side=tk.RIGHT
            )

    # ========================================================
    # 3D PLOT
    # ========================================================

    def create_3d_plot(self):

        self.fig3d = Figure(
            figsize=(8, 6),
            dpi=100
        )

        self.ax3d = self.fig3d.add_subplot(
            111,
            projection="3d"
        )

        self.canvas3d = FigureCanvasTkAgg(
            self.fig3d,
            master=self.tab_3d
        )

        self.canvas3d.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

    # ========================================================
    # 2D PLOT
    # ========================================================

    def create_2d_plot(self):

        control_frame = ttk.LabelFrame(
            self.tab_2d,
            text="2D Plot Settings",
            padding=10
        )

        control_frame.pack(
            fill=tk.X,
            padx=10,
            pady=10
        )

        ttk.Label(
            control_frame,
            text="Plot variable:"
        ).grid(
            row=0,
            column=0,
            padx=(0, 10),
            pady=5,
            sticky="w"
        )

        variable_combo = ttk.Combobox(
            control_frame,
            textvariable=self.two_d_variable_var,
            values=[
                "Length",
                "Width"
            ],
            state="readonly",
            width=20
        )

        variable_combo.grid(
            row=0,
            column=1,
            padx=5,
            pady=5,
            sticky="w"
        )

        variable_combo.bind(
            "<<ComboboxSelected>>",
            self.two_d_variable_changed
        )

        self.fixed_point_label = ttk.Label(
            control_frame
        )

        self.fixed_point_label.grid(
            row=1,
            column=0,
            padx=(0, 10),
            pady=5,
            sticky="w"
        )

        self.fixed_point_entry = ttk.Entry(
            control_frame,
            textvariable=self.fixed_point_var,
            width=20
        )

        self.fixed_point_entry.grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky="w"
        )

        self.fixed_point_unit = ttk.Label(
            control_frame,
            text="[cm]"
        )

        self.fixed_point_unit.grid(
            row=1,
            column=2,
            padx=5,
            pady=5,
            sticky="w"
        )

        ttk.Button(
            control_frame,
            text="Update Plot",
            command=self.update_2d_from_input
        ).grid(
            row=0,
            column=3,
            rowspan=2,
            padx=20,
            pady=5
        )

        self.fig2d = Figure(
            figsize=(8, 6),
            dpi=100
        )

        self.ax2d = self.fig2d.add_subplot(
            111
        )

        self.canvas2d = FigureCanvasTkAgg(
            self.fig2d,
            master=self.tab_2d
        )

        self.canvas2d.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self.update_fixed_point_label()

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def get_inputs(self):

        try:

            l_min = float(
                self.l_min_var.get()
            )

            l_max = float(
                self.l_max_var.get()
            )

            l_samples = int(
                self.l_samples_var.get()
            )

            w_min_mm = float(
                self.w_min_var.get()
            )

            w_max_mm = float(
                self.w_max_var.get()
            )

            w_samples = int(
                self.w_samples_var.get()
            )

            t_mm = float(
                self.t_var.get()
            )

        except ValueError:

            raise ValueError(
                "Please enter valid numerical values."
            )

        if not np.isfinite(
            [
                l_min,
                l_max,
                w_min_mm,
                w_max_mm,
                t_mm
            ]
        ).all():

            raise ValueError(
                "All numerical values must be finite."
            )

        if l_min <= 0:

            raise ValueError(
                "Length minimum must be greater than zero."
            )

        if l_max <= 0:

            raise ValueError(
                "Length maximum must be greater than zero."
            )

        if w_min_mm <= 0:

            raise ValueError(
                "Width minimum must be greater than zero."
            )

        if w_max_mm <= 0:

            raise ValueError(
                "Width maximum must be greater than zero."
            )

        if t_mm <= 0:

            raise ValueError(
                "Copper thickness must be greater than zero."
            )

        if l_max <= l_min:

            raise ValueError(
                "Length maximum must be greater than "
                "length minimum."
            )

        if w_max_mm <= w_min_mm:

            raise ValueError(
                "Width maximum must be greater than "
                "width minimum."
            )

        if l_samples < 2:

            raise ValueError(
                "Length samples must be at least 2."
            )

        if w_samples < 2:

            raise ValueError(
                "Width samples must be at least 2."
            )

        return (
            l_min,
            l_max,
            l_samples,
            w_min_mm,
            w_max_mm,
            w_samples,
            t_mm
        )

    # ========================================================
    # FIXED POINT VALIDATION
    # ========================================================

    def get_fixed_point(self):

        try:

            fixed_point = float(
                self.fixed_point_var.get()
            )

        except ValueError:

            raise ValueError(
                "Please enter a valid fixed-point value."
            )

        if not np.isfinite(
            fixed_point
        ):

            raise ValueError(
                "The fixed point must be finite."
            )

        if fixed_point <= 0:

            raise ValueError(
                "The fixed point must be greater than zero."
            )

        if self.two_d_variable_var.get() == "Length":

            # Fixed width is entered in mm.

            w_min_mm = float(
                self.w_min_var.get()
            )

            w_max_mm = float(
                self.w_max_var.get()
            )

            if not (
                w_min_mm
                <= fixed_point
                <= w_max_mm
            ):

                raise ValueError(
                    f"Fixed width must be between "
                    f"{w_min_mm:g} mm and "
                    f"{w_max_mm:g} mm."
                )

        else:

            # Fixed length is entered in cm.

            l_min = float(
                self.l_min_var.get()
            )

            l_max = float(
                self.l_max_var.get()
            )

            if not (
                l_min
                <= fixed_point
                <= l_max
            ):

                raise ValueError(
                    f"Fixed length must be between "
                    f"{l_min:g} cm and "
                    f"{l_max:g} cm."
                )

        return fixed_point

    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(self):

        try:

            values = self.get_inputs()

            (
                l_min,
                l_max,
                l_samples,
                w_min_mm,
                w_max_mm,
                w_samples,
                t_mm
            ) = values

            (
                self.length_grid,
                self.width_grid,
                self.inductance
            ) = calculate_inductance(
                l_min,
                l_max,
                l_samples,
                w_min_mm,
                w_max_mm,
                w_samples,
                t_mm
            )

            self.update_3d_plot()

            self.update_2d_from_input(
                show_error=False
            )

            self.status_var.set(
                "Calculation complete."
            )

        except ValueError as error:

            messagebox.showerror(
                "Input Error",
                str(error)
            )

    # ========================================================
    # 3D ASPECT RATIO
    # ========================================================

    def set_3d_aspect_ratio(self):

        # ----------------------------------------------------
        # Both axes internally in cm.
        # ----------------------------------------------------

        width_min_cm = (
            float(
                self.w_min_var.get()
            )
            / 10.0
        )

        width_max_cm = (
            float(
                self.w_max_var.get()
            )
            / 10.0
        )

        length_min_cm = float(
            self.l_min_var.get()
        )

        length_max_cm = float(
            self.l_max_var.get()
        )

        width_range = (
            width_max_cm
            - width_min_cm
        )

        length_range = (
            length_max_cm
            - length_min_cm
        )

        width_range = max(
            width_range,
            1e-12
        )

        length_range = max(
            length_range,
            1e-12
        )

        largest = max(
            width_range,
            length_range
        )

        x_ratio = (
            width_range
            / largest
        )

        y_ratio = (
            length_range
            / largest
        )

        # ----------------------------------------------------
        # Minimum 1:2 visual ratio.
        #
        # Prevents weird aspects
        # ----------------------------------------------------

        minimum_ratio = 0.5

        x_ratio = max(
            x_ratio,
            minimum_ratio
        )

        y_ratio = max(
            y_ratio,
            minimum_ratio
        )

        z_ratio = 0.6

        self.ax3d.set_box_aspect(
            (
                x_ratio,
                y_ratio,
                z_ratio
            )
        )

        self.ax3d.set_xlim(
            width_min_cm,
            width_max_cm
        )

        self.ax3d.set_ylim(
            length_min_cm,
            length_max_cm
        )

        # ----------------------------------------------------
        # Z limits based only on valid points.
        # ----------------------------------------------------

        valid_L = self.inductance.compressed()

        if valid_L.size > 0:

            z_min = float(
                np.min(valid_L)
            )

            z_max = float(
                np.max(valid_L)
            )

            if z_max <= z_min:

                z_max = z_min + 1.0

            self.ax3d.set_zlim(
                z_min,
                z_max
            )

    # ========================================================
    # UPDATE 3D PLOT
    # ========================================================

    def update_3d_plot(self):

        # ----------------------------------------------------
        # Remove old colorbar.
        # ----------------------------------------------------

        if self.colorbar is not None:

            try:

                self.colorbar.remove()

            except Exception:

                pass

            self.colorbar = None

        # ----------------------------------------------------
        # Clear axes.
        # ----------------------------------------------------

        self.ax3d.clear()


        self.surface = self.ax3d.plot_surface(
            self.width_grid,
            self.length_grid,
            self.inductance,
            cmap=self.cmap_var.get(),
            norm=mcolors.PowerNorm(
                gamma=0.75
            ),
            linewidth=0,
            antialiased=True
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.ax3d.set_xlabel(
            "Width [cm]"
        )

        self.ax3d.set_ylabel(
            "Length [cm]"
        )

        self.ax3d.set_zlabel(
            "Inductance [nH]"
        )

        self.ax3d.set_title(
            "PCB Trace Inductance - "
            "Terman Formula"
        )

        # ----------------------------------------------------
        # Aspect ratio and limits.
        # ----------------------------------------------------

        self.set_3d_aspect_ratio()


        self.colorbar = self.fig3d.colorbar(
            self.surface,
            ax=self.ax3d,
            shrink=0.7,
            aspect=20,
            pad=0.1
        )

        self.colorbar.set_label(
            "Inductance [nH]"
        )

        self.fig3d.tight_layout()

        self.canvas3d.draw()

    # ========================================================
    # 2D VARIABLE CHANGED
    # ========================================================

    def two_d_variable_changed(
        self,
        event=None
    ):

        self.update_fixed_point_label()

        if (
            self.two_d_variable_var.get()
            == "Length"
        ):

            # Fixed width is entered in mm.

            self.fixed_point_var.set(
                str(
                    DEFAULTS["fixed_width"]
                )
            )

        else:

            # Fixed length is entered in cm.

            self.fixed_point_var.set(
                str(
                    DEFAULTS["fixed_length"]
                )
            )

        self.update_2d_from_input()

    # ========================================================
    # FIXED POINT LABEL
    # ========================================================

    def update_fixed_point_label(self):

        if (
            self.two_d_variable_var.get()
            == "Length"
        ):

            self.fixed_point_label.config(
                text="Fixed width:"
            )

            self.fixed_point_unit.config(
                text="[mm]"
            )

        else:

            self.fixed_point_label.config(
                text="Fixed length:"
            )

            self.fixed_point_unit.config(
                text="[cm]"
            )

    # ========================================================
    # UPDATE 2D FROM INPUT
    # ========================================================

    def update_2d_from_input(
        self,
        show_error=True
    ):

        if self.length_grid is None:

            return

        try:

            fixed_point = (
                self.get_fixed_point()
            )

            self.update_2d_plot(
                fixed_point
            )

        except ValueError as error:

            if show_error:

                messagebox.showerror(
                    "2D Plot Error",
                    str(error)
                )

    # ========================================================
    # UPDATE 2D PLOT
    # ========================================================

    def update_2d_plot(
        self,
        fixed_point
    ):

        self.ax2d.clear()

        variable = (
            self.two_d_variable_var.get()
        )


        t_cm = (
            float(
                self.t_var.get()
            )
            / 10.0
        )

        # ====================================================
        # INDUCTANCE VS LENGTH
        # ====================================================

        if variable == "Length":


            fixed_width_cm = (
                fixed_point / 10.0
            )

            lengths_cm = np.linspace(
                float(
                    self.l_min_var.get()
                ),
                float(
                    self.l_max_var.get()
                ),
                int(
                    self.l_samples_var.get()
                )
            )

            # ------------------------------------------------
            # Terman Formula
            # ------------------------------------------------

            inductance = (
                2.0
                * lengths_cm
                * (
                    np.log(
                        (2.0 * lengths_cm)
                        / (
                            fixed_width_cm
                            + t_cm
                        )
                    )
                    + 0.5
                    + 0.2235
                    * (
                        (
                            fixed_width_cm
                            + t_cm
                        )
                        / lengths_cm
                    )
                )
            )

            # ------------------------------------------------
            # Mask:
            #
            #     L > 3W
            #
            # ------------------------------------------------

            valid = (
                lengths_cm
                > (
                    3.0
                    * fixed_width_cm
                )
            )

            inductance_plot = np.ma.masked_where(
                ~valid,
                inductance
            )

            self.ax2d.plot(
                lengths_cm,
                inductance_plot,
                linewidth=2,
                color="#ff7f0e"
            )

            self.ax2d.set_xlabel(
                "Length [cm]"
            )

            self.ax2d.set_title(
                "Inductance vs Length"
            )

            self.ax2d.text(
                0.02,
                0.95,
                f"Fixed width = "
                f"{fixed_point:g} mm",
                transform=self.ax2d.transAxes,
                verticalalignment="top"
            )

        # ====================================================
        # INDUCTANCE VS WIDTH
        # ====================================================

        else:


            fixed_length_cm = fixed_point


            widths_mm = np.linspace(
                float(
                    self.w_min_var.get()
                ),
                float(
                    self.w_max_var.get()
                ),
                int(
                    self.w_samples_var.get()
                )
            )

            widths_cm = (
                widths_mm / 10.0
            )

            # ------------------------------------------------
            # Terman Formula
            # ------------------------------------------------

            inductance = (
                2.0
                * fixed_length_cm
                * (
                    np.log(
                        (
                            2.0
                            * fixed_length_cm
                        )
                        / (
                            widths_cm
                            + t_cm
                        )
                    )
                    + 0.5
                    + 0.2235
                    * (
                        (
                            widths_cm
                            + t_cm
                        )
                        / fixed_length_cm
                    )
                )
            )

            # ------------------------------------------------
            # Mask:
            #
            #     L > 3W
            #
            # ------------------------------------------------

            valid = (
                fixed_length_cm
                > (
                    3.0
                    * widths_cm
                )
            )

            inductance_plot = np.ma.masked_where(
                ~valid,
                inductance
            )

            self.ax2d.plot(
                widths_mm,
                inductance_plot,
                linewidth=2,
                color="#ff7f0e"
            )

            self.ax2d.set_xlabel(
                "Width [mm]"
            )

            self.ax2d.set_title(
                "Inductance vs Width"
            )

            self.ax2d.text(
                0.02,
                0.95,
                f"Fixed length = "
                f"{fixed_length_cm:g} cm",
                transform=self.ax2d.transAxes,
                verticalalignment="top"
            )

        # ----------------------------------------------------
        # Common formatting
        # ----------------------------------------------------

        self.ax2d.set_ylabel(
            "Inductance [nH]"
        )

        self.ax2d.grid(
            True,
            alpha=0.3
        )

        self.ax2d.margins(
            x=0.03,
            y=0.08
        )

        self.fig2d.tight_layout()

        self.canvas2d.draw()

    # ========================================================
    # RESET
    # ========================================================

    def reset_defaults(self):

        self.l_min_var.set(
            str(
                DEFAULTS["l_min"]
            )
        )

        self.l_max_var.set(
            str(
                DEFAULTS["l_max"]
            )
        )

        self.l_samples_var.set(
            str(
                DEFAULTS["l_samples"]
            )
        )

        self.w_min_var.set(
            str(
                DEFAULTS["w_min"]
            )
        )

        self.w_max_var.set(
            str(
                DEFAULTS["w_max"]
            )
        )

        self.w_samples_var.set(
            str(
                DEFAULTS["w_samples"]
            )
        )

        self.t_var.set(
            str(
                DEFAULTS["t_mm"]
            )
        )

        self.fixed_point_var.set(
            str(
                DEFAULTS["fixed_width"]
            )
        )

        self.cmap_var.set(
            DEFAULTS["cmap"]
        )

        self.two_d_variable_var.set(
            "Length"
        )

        self.update_fixed_point_label()

        self.status_var.set(
            "Defaults restored."

        )

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    def save_graph(self):

        current_tab = (
            self.notebook.index(
                self.notebook.select()
            )
        )

        if current_tab == 0:

            figure = self.fig3d
            graph_type = "3D"

        else:

            figure = self.fig2d
            graph_type = "2D"

        # ----------------------------------------------------
        # Automatic filename
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        filename = (
            "PCB_Trace_Inductance_"
            f"{graph_type}_"
            f"{timestamp}.png"
        )

        filepath = (
            filedialog.asksaveasfilename(
                title="Save Graph",
                initialfile=filename,
                defaultextension=".png",
                filetypes=[
                    (
                        "PNG image",
                        "*.png"
                    )
                ]
            )
        )

        if not filepath:

            return

        try:

            figure.savefig(
                filepath,
                dpi=600,
                bbox_inches="tight"
            )

            self.status_var.set(
                f"Saved: {filepath}"
            )

            messagebox.showinfo(
                "Graph Saved",
                f"Graph successfully saved:\n\n"
                f"{filepath}"
            )

        except Exception as error:

            messagebox.showerror(
                "Save Error",
                f"Could not save graph:\n\n"
                f"{error}"
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = PCBTraceInductanceApp(
        root
    )

    root.mainloop()
