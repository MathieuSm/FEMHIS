#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script collects results from the statistical
    analysis, write and build a corresponding report
    using pylatex

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2023
    """

#%% Imports
# Modules import

import os
import argparse
from pathlib import Path
from Utils import SetDirectories

from pylatex import Document, Section, Subsection, Subsubsection
from pylatex import Figure, SubFigure, NoEscape, NewPage, Command
from pylatex.package import Package

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEMHIS')
    SD = RD / 'Statistics'

    Folders = sorted([F for F in Path.iterdir(SD) if Path.is_dir(F)])
    Sections = ['Cement Lines', 'Haversian Canals', 'Osteocytes']

    # Generate report
    Doc = Document(default_filepath=str(RD / 'Results'))
    for iF, Folder in enumerate(Folders):

        # Create section for tissue type
        with Doc.create(Section(Sections[iF], numbering=False)):

            Variables = sorted([F[:-10] for F in os.listdir(Folder) if F.endswith('.tex')])

            # Create subsections for each correlated variable
            for V in Variables:

                with Doc.create(Subsection(V.replace('_',' '), numbering=False)):  

                    # Add LME results (Figure and Table)
                    with Doc.create(Figure(position='h!')) as Fig:
                        Image = str(Folder / (V + '_LME.png'))
                        Fig.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                        Fig.add_caption(NoEscape('LME results'))

                    with open(str(Folder / (V + '_Table.tex'))) as Table:
                        Tex = Table.read()
                    Tex = Tex[:13] + '[h!]' + Tex[13:]
                    Doc.append(NoEscape(Tex))
                    Doc.append(NewPage())

                    if V == Variables[0]:

                        # Check assumptions: Density normal distribution
                        with Doc.create(Figure(position='h!')) as Fig:
                            Doc.append(Command('centering'))
                            
                            SubFig = SubFigure(position='b', width=NoEscape(r'0.55\linewidth'))
                            with Doc.create(SubFig) as SF:
                                Doc.append(Command('centering'))
                                Image = str(Folder / 'Density_Hist.png')
                                SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                                SF.add_caption('Histogram')

                            SubFig = SubFigure(position='b', width=NoEscape(r'0.45\linewidth'))
                            with Doc.create(SubFig) as SF:
                                Doc.append(Command('centering'))
                                Image = str(Folder / 'Density_QQ.png')
                                SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                                SF.add_caption('QQ plot')

                        Fig.add_caption('Density distribution')

                    # Check assumptions: X normal distribution
                    with Doc.create(Figure(position='h!')) as Fig:
                        Doc.append(Command('centering'))
                        
                        SubFig = SubFigure(position='b', width=NoEscape(r'0.55\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Hist.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('Histogram')

                        SubFig = SubFigure(position='b', width=NoEscape(r'0.45\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_QQ.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('QQ plot')

                    Fig.add_caption('Independent variable distribution')
                    Doc.append(NewPage())

                    # Check assumptions: Random effect normal distribution and 0 mean
                    with Doc.create(Figure(position='h!')) as Fig:
                        Doc.append(Command('centering'))
                        
                        SubFig = SubFigure(position='t', width=NoEscape(r'0.34\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_RE.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('Boxplot')

                        SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Group_QQ.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('Group effect QQ plot')
                    
                    with Doc.create(Figure(position='h!')) as Fig:
                        Doc.append(Command('centering'))
                        
                        SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Left_QQ.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('Sample effect: Left QQ plot')

                        SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Right_QQ.png'))
                            SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
                            SF.add_caption('Sample effect: Right QQ plot')

                    Fig.add_caption('Random effects distribution and 0 mean assumptions')

                    # Check assumptions: Residuals distribution and 0 mean
                    with Doc.create(Figure(position='h!')) as Fig:
                        Doc.append(Command('centering'))

                        SubFig = SubFigure(position='b', width=NoEscape(r'0.27\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Residuals.png'))
                            SF.add_image(Image, width=NoEscape(r'0.8\linewidth'))
                            SF.add_caption('Boxplot')

                        SubFig = SubFigure(position='b', width=NoEscape(r'0.5\linewidth'))
                        with Doc.create(SubFig) as SF:
                            Doc.append(Command('centering'))
                            Image = str(Folder / (V + '_Residuals_QQ.png'))
                            SF.add_image(Image, width=NoEscape(r'0.8\linewidth'))
                            SF.add_caption('QQ plot')

                    Fig.add_caption('Residuals distribution and 0 mean assumptions')

                Doc.append(NewPage())

    Doc.packages.append(Package('caption', options='labelformat=empty'))
    Doc.packages.append(Package('subcaption', options='aboveskip=0pt, labelformat=empty'))
    Doc.generate_pdf(clean_tex=False)   

#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
# %%
