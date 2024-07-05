import scipdf
import os
from typing import Any, List
import json
import fitz
import pickle as pkl


def extract_formula_image(pdf_path: str, coords: List, output_image_path: str, dpi=500):
    doc = fitz.open(pdf_path)  # Open the PDF
    page_number = int(coords[0]) - 1  # Adjust the page number for 0-indexing
    page = doc.load_page(page_number)  # Load the specified page

    # The coordinates are interpreted as [page_number, x0, y0, width, height]
    # Convert to a rectangle using [x0, y0, x0+width, y0+height] for extraction
    rect = fitz.Rect(coords[1], coords[2], coords[1] + coords[3], coords[2] + coords[4])

    # Crop the page to the desired area
    # Increase the resolution by setting the dpi parameter
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # A scaling matrix for the desired DPI
    pix = page.get_pixmap(matrix=mat, clip=rect)

    # Save the extracted image
    pix.save(output_image_path)


def get_paper_info(pdf_path: str = None, figure_path: str = 'scipdf_figures') -> [Any]:
    """Given pdf path, find the paper title, author information
    and other info."""
    pdf_name = pdf_path.split('/')[-1]
    print('pdf_name is ', pdf_name)

    # TODO: skip scipdf reparsing again, the caller might already parsed
    article_dict = scipdf.parse_pdf_to_dict(pdf_path)

    # scipdf.parse_figures(pdf_path, output_folder=figure_path)
    # print("article_dict:", article_dict)

    title = article_dict['title']
    author = article_dict['authors']
    abstract = article_dict['abstract']
    sections = article_dict['sections']
    figures = article_dict['figures']
    pdf_folder = os.path.dirname(pdf_path)
    scipdf.parse_figures(pdf_folder, output_folder=figure_path)
    real_figures = {}
    real_tables = {}
    json_path = os.path.join(figure_path, 'data', pdf_name.split('.pdf')[0]+'.json')
    json_data = open(json_path, 'r').readlines()
    json_data = ''.join(json_data)
    json_data = json.loads(json_data)
    assert type(json_data) == type([])
    for i in range(len(json_data)):
        this_data_name = str(json_data[i]['name'])
        this_data_type = json_data[i]['figType']
        if this_data_type == 'Figure':
            real_figures[this_data_name] = {
                'fig_path': json_data[i]['renderURL'],
                'caption': json_data[i]['caption']
            }
        elif this_data_type == 'Table':
            real_tables[this_data_name] = {
                'table_path': json_data[i]['renderURL'],
                'caption': json_data[i]['caption']
            }
    
    # formulas
    all_formulas = {}
    os.makedirs(figure_path+'/formulas', exist_ok=True)
    for i in range(len(article_dict['formulas'])):
        this_formula = article_dict['formulas'][i]
        formula_name = this_formula['formula_id']
        formula_info = this_formula['formula_text']
        formula_coord = this_formula['formula_coordinates']
        formula_coord[0] = int(formula_coord[0])
        formula_path = os.path.join(figure_path, 'formulas', formula_name+'.png')
        all_formulas[formula_name] = {'formula_text': formula_info, 'formula_path': formula_path, 'formula_page': formula_coord[0], 'formula_coord': formula_coord[1:]}
        extract_formula_image(pdf_path, formula_coord, formula_path)
    return article_dict, title, author, abstract, sections, figures, all_formulas, real_figures, real_tables


if __name__ == '__main__':
    pdf_path = 'https://arxiv.org/pdf/2404.06731.pdf'
    figure_path = 'scipdf_figures/motion'
    article_dict, title, author, abstract, sections, figures, all_formulas, real_figures, real_tables = get_paper_info(pdf_path, figure_path)
    all_info = [article_dict, title, author, abstract, sections, figures, all_formulas, real_figures, real_tables]
    os.makedirs('info_data', exist_ok=True)
    pkl.dump(all_info, open('info_data/all_info_'+pdf_path.split('/')[-1].split('.pdf')[0]+'.pkl', 'wb'))
