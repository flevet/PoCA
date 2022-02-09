# -*- coding: utf-8 -*-
'''
Various supporting functions

@author: Dave

'''
import platform
import itertools
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import smtplib
from email.mime.text import MIMEText
import math
import matplotlib as mpl
import os
import ctypes

def send_jobdone_email(strJobID, strSubject = ' Processing Complete', extraInfo = None):
    '''
    Send notification emails when called, e.g. when a long job has finished.
    You need to specify valid settings in the file 'email_secrets.json'
    for it to work.
    
    It is advisable to set up a free email account somewhere for this purpose.
    
    It is INADVISABLE to use your personal, private, or institutional email account!
    
    You must specify values for

        sender_name          The name for the sending account (e.g. 'CAML Notifier')
        sender_email_addr    The email address for the sending account (e.g. noreply@outgoingserver.com)
        sender_login         The sending account's login name
        sender_passwd        The sending account's login password
        sender_host_name     The sending mail-server's hostname (e.g. mail.outgoingserver.com)
        sender_host_port     The sending mail-server's port (e.g. 587)
        recipient_email_addr The email address which will receive notification emails (e.g. perinimankatz@gmail.com)
        recipient_name       The name to use in the notification emails (e.g. 'Perini Mankatz')
        
    in the file 'email_secrets.json'. An example of this file is distributed as 
    'email_secrets_DIST.json'. Open this file, save it as 'email_secrets.json',
    then modify the settings and save it again.

    '''

    # check for a valid emai-settings JSON file and load it
    email_secrets_json = 'email_secrets.json'
    if os.path.isfile(email_secrets_json):
        with open(email_secrets_json, 'r') as file:
            email_secrets = json.loads(file.read())
            
        if not all(('sender_name' in email_secrets,          \
                   'sender_email_addr' in email_secrets ,   \
                   'sender_login' in email_secrets ,        \
                   'sender_passwd' in email_secrets ,       \
                   'sender_host_name' in email_secrets ,    \
                   'sender_host_port' in email_secrets ,    \
                   'recipient_email_addr' in email_secrets, \
                   'recipient_name' in email_secrets)):
            return(warn_msg('Email notifier: settings file found but does not contain valid settings!\r\n'+
                 'Email notifications cannot be sent.'))
    else:
        return(warn_msg('Email notifier: Settings file (\'email_secrets.json\') could not be found!\r\n'+
                 'Email notifications cannot be sent.'))

    # message construction
    msg_subject = 'CAML - ' + strSubject
    msg_body_start = 'Hello!\r\nProcessing has been completed on \'%s\' for the job below.\r\n\r\n%s\r\n' % (os.uname()[1], strJobID)
    msg_body_end = '\r\nThanks!\r\n'
    
    if extraInfo:
        msg_body = msg_body_start + extraInfo + msg_body_end
    else:
        msg_body = msg_body_start + msg_body_end
    
    # build email
    message = MIMEText(msg_body)
    message['Subject'] = msg_subject
    message['From'] = email_secrets['sender_name'] + '<' + email_secrets['sender_email_addr'] + '>'
    message['To'] = email_secrets['recipient_name'] + '<' + email_secrets['recipient_email_addr'] + '>'

    try:
        smtpObj = smtplib.SMTP(email_secrets['sender_host_name'], email_secrets['sender_host_port'])
        smtpObj.connect(email_secrets['sender_host_name'], email_secrets['sender_host_port'])
        smtpObj.ehlo()
        smtpObj.starttls()
        smtpObj.ehlo()
        smtpObj.login(email_secrets['sender_login'], email_secrets['sender_passwd'])
        smtpObj.sendmail(email_secrets['sender_email_addr'], email_secrets['recipient_email_addr'], message.as_string())
        smtpObj.quit()
        return 'Notification message sent to ' + email_secrets['recipient_email_addr']
    
    except:
        return(warn_msg('Error: Notification message could not be sent\r\n' + 
               'Check the settings in the \'email_secrets.json\' file.'))


def askforinput(message, errormessage, defaultval, isvalid):
    '''
    Prompt for input given a message and return that value after verifying the input.

    Keyword arguments:
    message -- the message to display when asking the user for the value
    errormessage -- the message to display when the value fails validation
    defaultval -- the default value if not input is given
    isvalid -- a function that returns True if the value given by the user is valid
    '''
    validresult = False

    # if a default value is given, check that it is sensible   
    if defaultval is not None and not isvalid(defaultval):
        defaultval = None
    
    while not validresult:
        
        if defaultval is not None:
            result = input(fancy(str(message), 'cyan','darkgrey') + fancy(' [ Default is ' + defaultval + ' ]', 'white','darkgrey') + ': ')
        else:
            result = input(fancy(str(message), 'cyan','darkgrey') + ': ')
        
        # no input given but we have a default option
        if not result and defaultval is not None:
            result = defaultval
            print(fancy('Using default:', 'black', 'green') + ' ' + result + '\n')
            validresult = True

        else:
            # input was supplied
            
            # Windows can get trapped at these prompts as Spyder ignores Ctrl-X
            # we can capture input and cancel running here instead
            if result=='exit!':
                raise ValueError('You entered \'exit!\' so the script has stopped running.')
            
            # strip 'file:///' from Windows copy-paste input
            if result.startswith('file:///'):
                result = result[len('file:///'):]
            
            # check for badness
            if not isvalid(result):
                print(fancy(str(errormessage), 'yellow', 'black'), end='\n\n')
                result = None
            # input was supplied and it was good
            else:
                print(fancy('Acknowledged:', 'black', 'green') + ' ' + result + '\n')
                validresult = True            
    return result


def fancy(inputstr, fg, bg, flip=False):
    '''
    Returns inputstr surrounded by ANSI escape sequences. Printing outputstr will
    render the string in the corresponding foreground and background colours.
    This can make it easier to highlight messages or action dialog in the console.
    '''

    colors = {
            'black' : '0',
            'darkred' : '1',
            'darkgreen' : '2',
            'olive' : '3',
            'dark_blue' : '4',
            'purple' : '5',
            'teal' : '6',
            'lightgrey' : '7',
            'darkgrey' : '8',
            'red' : '9',
            'green' : '10',
            'yellow' : '11',
            'blue' : '12',
            'magenta' : '13',
            'cyan' : '14',
            'white' : '15'
            }
    
    # if either of the input colors aren't in the list above then you just get
    # white text on black background
    try:
        if flip:
            fg = colors[bg]
            bg = colors[fg]            
        else:
            fg = colors[fg]
            bg = colors[bg]
    except:
        if flip:
            fg = '0'  # black text
            bg = '15' # white background
        else:
            fg = '15' # white text
            bg = '0'  # black background

    # build the output string:
    #    \x1b[       escape character in Hex ASCII
    #    38;         38 to set foreground colour or 48 to set background colour
    #    5;          use 8-bit lookup table
    #    fg; or bg;  color index in the lookup table
    #    \x1b[0m     reset color to the default / remove all attributes and m = finish the escape sequence
    outputstr = '\x1b[38;5;' + fg + ';48;5;' + bg + 'm' + inputstr + '\x1b[0m'
    return outputstr


def ok_msg(msg_str):
    print('\x1b[1;30;42m' + '     OK!    ' + '\x1b[0m' + '  ' + msg_str, flush=True)
        
    
def err_msg(msg_str):
    print('\n\n\x1b[1;35;43m' + '  PROBLEM!  ' + '\x1b[1;37;45m' + '  ' + msg_str + '  \x1b[0m\n\n', flush=True)


def info_msg(msg_str):
    print('\x1b[1;37;44m' + '    INFO    ' + '\x1b[0m' + '  ' + msg_str, flush=True)


def warn_msg(msg_str):
    print('\x1b[1;30;43m' + '  WARNING!  ' + '\x1b[0m' + '  ' + msg_str, flush=True)


def progress_msg(prog_ctr, msg_str):
    padding_str = (int(len(prog_ctr) / 2)) * ' '
    print('\n\x1b[1;37;44m' + padding_str + prog_ctr + padding_str + '\x1b[1;30;47m' + '  ' + msg_str + '  \x1b[0m', flush=True)

    
def complete_msg():
    print('\x1b[36;1m' + ' OK! ' + '\x1b[0m', flush=True)


def make_image(inputdata, outputname, ps_input, colormapID):
    '''
    Creates images of the same size, format etc.
    '''
    
    if 'OutputScaling' in ps_input:
        OutputScaling = ps_input['OutputScaling']
    else:
        if np.log10(max((ps_input['xMax'],ps_input['yMax']))) < 3:
            OutputScaling = 1.0  # nm per pixel
        else:
            OutputScaling = 100.0  # nm per pixel
        
    if 'PointSize' in ps_input:
        PointSize = ps_input['PointSize']
    else:
        PointSize = 1.0
    
    if 'BackGroundColor' in ps_input:
        BackGroundColor = ps_input['BackGroundColor']
    else:
        BackGroundColor = (1, 1, 1) #(0.1765, 0.1765, 0.1765)
        
    if 'ForeGroundColor' in ps_input:
        ForeGroundColor = ps_input['ForeGroundColor']
    else:
        ForeGroundColor = (0, 0, 0)
    # MinValColor = (0.5, 0.5, 0.5)
    
    # Update for Matplotlib 3.1: every point needs its own color
    ForeGroundColor = np.full((inputdata.shape[0],3),ForeGroundColor)    
    
    if colormapID == None:
        colormapID = 'cool'
    
    # Fixed paper size, variable DPI
    try:
        saveHighDPI = ps_input['ImageSize'][0] / OutputScaling
    except:
        saveHighDPI = ps_input['ImageSize'] / OutputScaling
    
    if saveHighDPI > 3000:
        saveHighDPI = 300
    
    # saveHighDPI = ps_input['ImageSize'][0] / OutputScaling
    paperSize_x = 30.0 # 'inches'
    
    data_x_width = max(inputdata[:, ps_input['xCol']]) - min(inputdata[:, ps_input['xCol']])
    data_y_width = max(inputdata[:, ps_input['yCol']]) - min(inputdata[:, ps_input['yCol']])
    
    field_x_width = ps_input['xMax'] - ps_input['xMin']
    field_y_width = ps_input['yMax'] - ps_input['yMin']
    
    if (field_x_width / data_x_width) > 5 or (field_y_width / data_y_width) > 5:
        suggest_x_min = np.floor(np.min(inputdata[:,ps_input['xCol']]) / ps_input['AutoAxesNearest'] ) * ps_input['AutoAxesNearest']
        suggest_x_max = np.ceil(np.max(inputdata[:,ps_input['xCol']]) / ps_input['AutoAxesNearest'] ) * ps_input['AutoAxesNearest']
        suggest_y_min = np.floor(np.min(inputdata[:,ps_input['yCol']]) / ps_input['AutoAxesNearest'] ) * ps_input['AutoAxesNearest']
        suggest_y_max = np.ceil(np.max(inputdata[:,ps_input['yCol']]) / ps_input['AutoAxesNearest'] ) * ps_input['AutoAxesNearest']
        
        paperSize_y = ((suggest_y_max - suggest_y_min) / (suggest_x_max - suggest_x_min)) * paperSize_x
    else:
        paperSize_y = ((ps_input['yMax'] - ps_input['yMin']) / (ps_input['xMax'] - ps_input['xMin'])) * paperSize_x
       
    plt.ioff() # turn off interactive mode
    
    fig = plt.figure(frameon=False)
    fig.patch.set_facecolor(BackGroundColor)
    fig.set_facecolor(BackGroundColor)
    fig.set_size_inches(paperSize_x, paperSize_y)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax, frameon=True, facecolor=BackGroundColor)
    
    # nice plots from :
        # s=1, marker='*', edgecolors='none'  --> slightly aliased points
        # s=0.2, marker='s', edgecolors='none' --> square exact pixels
        # s=1, marker='*', edgecolors='none' --> very fine pixel points
    
    ColorByCluster = False
    
    if 'ClusMembershipIDCol' in ps_input:
        if ps_input['ClusMembershipIDCol'] >= 0 and type(ps_input['ClusMembershipIDCol']) == int:
            ColorByCluster = True

    if ColorByCluster:
        # background points (value 0) and colour others
        if 'PointsMinValue' in ps_input:
            PointsMinValue = ps_input['PointsMinValue']
        else:
            PointsMinValue = inputdata[:, ps_input['ClusMembershipIDCol']].min()

        if 'PointsMaxValue' in ps_input:
            PointsMaxValue = ps_input['PointsMaxValue']
        else:
            PointsMaxValue = inputdata[:, ps_input['ClusMembershipIDCol']].max()
            if PointsMaxValue == PointsMinValue:
                PointsMaxValue = PointsMinValue + 1
         
        MinValData = inputdata[np.where((inputdata[:, ps_input['ClusMembershipIDCol']] == PointsMinValue)), :][0,:,:]
        OtherData = inputdata[np.where((inputdata[:, ps_input['ClusMembershipIDCol']] != PointsMinValue)), :][0,:,:]
                
        # Plot the non-clustered data
        if MinValData.shape[0] > 0:
            ax.scatter(MinValData[:, ps_input['xCol']],
                       MinValData[:, ps_input['yCol']],
                       c=MinValData[:, ps_input['ClusMembershipIDCol']],
                       s=PointSize,
                       cmap=colormapID,
                       marker='.', 
                       edgecolors='none', 
                       vmin=PointsMinValue, 
                       vmax=PointsMaxValue,
                       zorder=-1)
            
        # plot the clustered data
        if OtherData.shape[0] > 0:
            ax.scatter(OtherData[:, ps_input['xCol']],
                       OtherData[:, ps_input['yCol']],
                       c=OtherData[:, ps_input['ClusMembershipIDCol']],
                       s=PointSize,
                       cmap=colormapID,
                       marker='.', 
                       edgecolors='none', 
                       vmin=PointsMinValue, 
                       vmax=PointsMaxValue,
                       zorder=1)
        
    else:
        # nothing to colour, make it all one color
        ax.scatter(inputdata[:, ps_input['xCol']],
                   inputdata[:, ps_input['yCol']],
                   s=PointSize,
                   c=ForeGroundColor,
                   marker='.',
                   edgecolors='none')
    
    if (field_x_width / data_x_width) > 5 or (field_y_width / data_y_width) > 5:
        plt.axis((suggest_x_min, suggest_x_max, suggest_y_min, suggest_y_max))        
    else:
        plt.axis((ps_input['xMin'],ps_input['xMax'],ps_input['yMin'],ps_input['yMax']))
        
    plt.rcParams['figure.figsize'] = [paperSize_x, paperSize_y]

    fig.savefig(outputname,
                dpi=saveHighDPI,
                bbox_inches=0,
                facecolor=fig.get_facecolor(),
                edgecolor='none',
                transparent=True)
    
    plt.close('all')

    plt.ion() # turn on interactive mode


def load_colormap(cmapfile, flip=False):
    '''
    Loads colormap from a text file.
    File formats are the same as for ImageJ LUTs, i.e. space-delimited text file of
    R G B values (in the range 0-255) with the first line being the minimum color
    and the last line the maximum color.
    '''
    
    if os.path.isfile(cmapfile):

        cm_data = np.loadtxt(cmapfile) # Load the file
                    
        if flip:
            cm_data = np.flip(cm_data, axis=0)
            
        # normalize values to (0,1]
        if np.amax(cm_data) > 1 or np.amin(cm_data) < 0:
            cm_data = cm_data/255.0
        
        try:
            customcolormap = mpl.colors.ListedColormap(cm_data)
            return customcolormap
        except:
            print('Unable to interpret your custom colormap file! Reverting to \'viridis\'')
            return 'viridis'
    
    else:
        
        builtins = [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                'twilight', 'twilight_shifted', 'hsv',
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                'tab10', 'tab20', 'tab20b', 'tab20c',
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        
        if cmapfile in builtins:
            return cmapfile
        else:
            print('Specified colormap is unavailable (not a built-in or the file does not exist)\r\n' +
                  'You gave: \''+ cmapfile +'\'\r\n' +
                  'Reverting to \'viridis\'')
            return 'viridis'

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    '''
    Generates a string of random uppercase letters and digits.
    size = length of the string
    chars = set of characters to draw from
    '''
    
    chars2use = chars
    if chars == 'mixed':
        chars2use = string.ascii_lowercase + string.ascii_uppercase + string.digits
    elif chars == 'lowercase':
        chars2use = string.ascii_lowercase
    elif chars == 'uppercase':
        chars2use = string.ascii_uppercase
    elif chars == 'numbers':
        chars2use = string.digits
        
    return ''.join(random.choice(chars2use) for _ in range(size))


def get_free_space(dirname):
    '''
    Return free space (in bytes) for the device containing dirname
    '''
    
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value
    else:
        statvfs = os.statvfs(dirname)
        free_space = (statvfs.f_frsize * statvfs.f_bavail)
        return free_space


def convert_size(size_bytes):
    '''
    Converts bytes to human-readable notation
    '''
    
    if size_bytes == 0:
        return '0B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '%s %s' % (s, size_name[i])


def plot_confusion_matrix(cm, 
                          classes,
                          cmap,
                          plottxtcolor,
                          plotbgcolor,
                          normalize=False,
                          title='Confusion matrix'):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        color_min = 0
        color_max = 1
        print('Normalized CM - \'' + title + '\'')
    else:
        color_min = 0
        color_max = np.max(np.sum(cm, axis=1))
        print('Absolute CM - \'' + title + '\'')

    print(cm)
    
    figCM = plt.figure(facecolor = plotbgcolor)
    figCM.add_subplot(111, facecolor = plotbgcolor)
    
    figCM.set_size_inches(10, 10)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=color_min, vmax=color_max)

    ax6 = plt.gca()
    ax6.tick_params(color=plottxtcolor, labelcolor=plottxtcolor)
    for spine in ax6.spines.values():
        spine.set_edgecolor(plottxtcolor)
        
    plt.title(title, color=plottxtcolor)
    plt.ylabel('True label', color=plottxtcolor)
    plt.xlabel('Predicted label', color=plottxtcolor)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.ax.minorticks_on()
    cbar.set_label('score', color=plottxtcolor)  # set colorbar label plus label color
    cbar.ax.yaxis.set_tick_params(color=plottxtcolor)     # set colorbar tick color
    cbar.outline.set_edgecolor(plottxtcolor)              # set colorbar edgecolor 
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=plottxtcolor)    

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, color=plottxtcolor)
    plt.yticks(tick_marks, classes, color=plottxtcolor)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color=plottxtcolor,
                 backgroundcolor=plotbgcolor)

    plt.tight_layout()
    return figCM


def multisplit(delimiters, string, maxsplit=0):
    '''
    This function splits a string across a list according to multiple delimiters.
    delimiters = (',', '_', '(', ')', 'x')
    string = 'Hello, this_is x something (very) interexsting'
    
    returns ['Hello', 'this', 'is', 'something', 'very', 'interesting']
    '''
    regexPattern = '|'.join(map(re.escape, delimiters))
    mulitsplit_tmp = re.split(regexPattern, string, maxsplit)
    mulitsplit_tmp = list(filter(None, mulitsplit_tmp))
    return mulitsplit_tmp


def duplicate_rows(array1, array2):
    '''
    Finds the rows in array1 which are also in array2
    arrays must be 2D and must match length in one dimension (same number of rows OR cols)
    Returns boolean same length as array1 indicating which rows are in array2
    source: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    '''
    
    tmp=np.prod(np.swapaxes(array1[:,:,None],1,2)==array2,axis=2)
    return np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)

