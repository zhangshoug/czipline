#!/usr/bin/python
""" Process notebook

Usage:
  process_notebook.py INPUT OUTPUT [--toc [POS] ] [--title TITLE | --between C1,C2]
  process_notebook.py INPUT -i [--toc [POS] ] [--title TITLE | --between C1,C2]

Options:
  -h --help         Show this screen.
  -i --inplace      Inplace modification of the notebook
  --toc  POS        Generate Table of Content, with optional position (default 1, or update existing)
  --title TITLE     Extract a section by title
  --between C1,C2   Extract a section by cell number
"""
import sys
import json
from docopt import docopt

def read_nb(fname):
    f = open(fname)
    nb = json.load(f)
    f.close()
    return nb

def write_nb(nb, fname):
    f = open(fname, 'w')
    json.dump(nb, f)
    f.close()

def filter_cells(cells=None, nb=None, worksheet=0, cell_type=None, custom=None, strip=True, **kwargs):
    """ filter notebook cells

    cells: list of cells (e.g. nb['worksheets'][0]['cells'])
    nb: notebook, if cells is not provided
    worksheet: worksheet number, if notebook if provided (default 0)
    cell_type
    custom: a function on cell returning True or False
    **kwargs: keyword arguements to test various cell fields
    """
    if cells is None:
        cells = nb['worksheets'][worksheet]['cells']

    # filtering function
    def filt(x):
        res = True
        if res and cell_type is not None:
            res = x['cell_type'] == cell_type
        for k in kwargs:
            if not res: break
            if strip:
                res = x[k].strip() == kwargs[k]
            else:
                res = x[k] == kwargs[k]

        if res and custom:
            res = custom(x)

        return res

    return filter(filt, cells)

def get_headings(nb, minlev=0, maxlev=100):
    #heading_cells = filter(lambda x: x['cell_type'] == 'heading', nb['worksheets'][0]['cells'])
    heading_cells = filter_cells(nb=nb, cell_type='heading', custom=lambda x: x['level'] >= minlev and x['level'] <= maxlev)
    headings = [cell['source'] for cell in heading_cells]
    levels = [cell['level'] for cell in heading_cells]
    return headings, levels

def get_group(nb, heading, worksheets=0):
    """ get all cells under a given heading
    """
    start = end = level = None
    for i, cell in enumerate(nb['worksheets'][worksheet]['cells']):

        # find the heading?
        if cell['cell_type'] =='heading' and cell['source'].strip() == heading:
            start = i
            level = cell['level']
            continue

        # stop when a heading of level equal or superior is found
        if start is not None and cell['cell_type'] =='heading' and cell['level'] >= level:
            end = i
            break

    if end is None: end = -1
    assert start is not None, "Header not found"

    cells = nb['worksheets'][worksheet]['cells'][start:end]
    return cells

def create_toc(headings, levels, minlev = 0, maxlev = 6):
    toc = []
    for i, hd in enumerate(headings):
        lev = int(levels[i])
        if lev > maxlev or lev < minlev:
            continue

        url = "#"+hd[0].replace(' ','-').replace('`','')
        entry = "    "*(lev-minlev) + "- [{}]({})".format(hd[0], url)
        toc.append(entry)
    return toc

def get_toc_cell(nb, FLAG, worksheet=0):
    """ return index of TOC cell
    """
    for i, cell in enumerate(nb['worksheets'][worksheet]['cells']):
        if cell['cell_type'] == 'markdown' and cell['source'][0].strip() == FLAG:
            return i, cell

    # toc not found, so just create it and insert it at the second position
    # create TOC cell
    cell = {
     "cell_type": "markdown",
     "metadata": {},
     }
    cell["source"] = toc
    cell['source'] = "\n".join([FLAG]+toc)

    i = 2

    return i, cell
    
def insert_toc(nb, toc, insert=None, worksheet=0):
    """ insert TABLE of content under the first header
    """
    TITLE='### Table of Content'

    cells = nb['worksheets'][worksheet]['cells']

    # check whether TOC is present
    found = False
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and cell['source'][0].strip() == TITLE:
            found = True
            break

    # toc not found, so just create it and insert it at the second position
    if not found:
        #import ipdb
        #ipdb.set_trace()
        print "TOC not found: create a new one"
        # create TOC cell
        cell = {
         "cell_type": "markdown",
         "metadata": {},
         }
        # insert at the second position
        if insert is None: insert = 1
        cells.insert(insert, cell)

    # move the TOC ?
    else:
        print "TOC found: update only"
        if insert is not None:
            print "move TOC"
            cell = cells.pop(i)
            cells.insert(insert, cell)

    # update TOC
    cell["source"] = [TITLE]+toc

def update_toc(nb, insert):
    # Create a table of content
    headings, levels = get_headings(nb)
    toc = create_toc(headings, levels, minlev=1)
    print "\n".join(toc)

    # Update TOC cell
    insert_toc(nb, toc, insert=insert)
    return nb

def extract_section_by_cells(nb, l1=None, l2=None):
    """
    """
    active = False
    if l1 is None:
        active = True

    # now just go throught the cells and extract the relevant ones
    cells = []
    for cell in nb['worksheets'][0]['cells']:
        if 'prompt_number' not in cell.keys():
            no = ''
        else:
            no = cell['prompt_number']

        # start copying?
        if not active and no == l1:
            active = True
        #else:
        #    print "no=",no,"and l1=",l1

        # copy cell if active
        if active:
            cells.append(cell)

        # stop copying?
        if l2 is not None and active and no == l2:
            active = False
            break

    if len(cells) == 0:
        import ipdb
        ipdb.set_trace()

    # update cells
    nb['worksheets'][0]['cells'] = cells

    return nb

def extract_section_by_title(nb, title):
    """ extract a section from a notebook, using title
    """
    active = False

    # now just go throught the cells and extract the relevant ones
    cells = []
    for cell in nb['worksheets'][0]['cells']:

        # stop copying? when a heading of same level is reached
        if active and cell['cell_type'] == 'heading' and cell['level'] == level:
            active = False
            break

        # start copying?
        if cell['cell_type'] == 'heading' and not active:
            searching = title.strip().lower()
            found = cell['source'][0].strip().lower()
            if searching==found and not active:
                active = True
                level = cell['level']
                print 'found title', title.strip(), ' with level', level
            #else:
            #    print 'searching:',searching, 'found:',found

        # copy cell if active
        if active:
            cells.append(cell)

    if len(cells) == 0:
        import ipdb
        ipdb.set_trace()

    # update cells
    nb['worksheets'][0]['cells'] = cells

    return nb

def main():

    arguments = docopt(__doc__)
    #print(arguments)
    #sys.exit()

    # File name
    nm_in = arguments['INPUT']

    if arguments['--inplace']:
        nm_out = nm_in
    else:
        nm_out = arguments['OUTPUT']

    nb = read_nb(fname=nm_in)

    # Extract a section?
    if arguments['--title']:
        nb = extract_section_by_title(nb, arguments['TITLE'])

    elif arguments['--between']:
        C1, C2 = [int(c) for c in arguments['C1,C2'].split(',')]
        nb = extract_section_by_cells(nb, C1, C2)

    # update TOC ?
    if arguments['--toc']:
        insert = arguments['POS']
        if insert is not None: insert = int(insert)
        nb = update_toc(nb, insert=insert)

    write_nb(nb, fname=nm_out)
    print 'Notebook written to', nm_out

if __name__ == '__main__':
    main()
