This folder contains data necessary for viability analysis of single-cell bacteria.
Bacterial species include E. Coli (EC: 1), Listeria (LI: 2), Staphy (SA: 3), and
Salmonella (SE: 4, SH: 5, ST, 6). This readme file briefly explains about the data.

single_cell_scinet.csv: Contains locations of source images and single-cell images, and their morphological and spectral data.
    BlobID: Unique ID for each single cell
    InImage_ID: hypercube-specific ID of single cell so that you can easily identify it in the hypercube
    Source_folder and _filename: location of 546nm band image of hypercubes.
    Out_folder and _filename: location of 546nm band image of the single cells.
    Area-Perimeter: These columns contains morphological properties. About the properties,
        refer to supplementary material of Kang et al. (2020) "Single-cell classification of foodborne pathogens
        using hyperspectral microscope imaging coupled with deep learning frameworks"
    Band_1-Band_299: These columns are raw spectrum of each cell as names imply
    Species: Contains Bacterial species (strains) ID as defined above
    Class_label: Indicates cell's viability; Live (1), Dead (2)

single_cells folder: Contains 546nm band images of individual single cells
