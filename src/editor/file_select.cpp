#include "file_select.hpp"

#ifdef USE_GTK
#include <gtk/gtk.h>
#include <glib.h>
#include <glib/gi18n.h>
#endif

namespace RLpbr {
namespace editor {

#ifdef USE_GTK

static const char *fileDialogGTK()
{
    gboolean success = gtk_init_check(nullptr, nullptr);
    if (success == FALSE) {
        return nullptr;
    }

    GtkWidget *dialog;
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    gint res;
    
    dialog = gtk_file_chooser_dialog_new("Open File",
                                         nullptr,
                                         action,
                                         _("_Cancel"),
                                         GTK_RESPONSE_CANCEL,
                                         _("_Open"),
                                         GTK_RESPONSE_ACCEPT,
                                         NULL);
    while (gtk_events_pending()) {
		gtk_main_iteration();
    }

    char *filename = nullptr;
    
    res = gtk_dialog_run(GTK_DIALOG(dialog));
    if (res == GTK_RESPONSE_ACCEPT) {
        char *g_filename;
        GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
        g_filename = gtk_file_chooser_get_filename(chooser);
        size_t num_chars = strlen(g_filename);
        filename = new char[num_chars + 1];
        filename[num_chars] = '\0';

        memcpy(filename, g_filename, num_chars);

        g_free (g_filename);
    }
    
    gtk_widget_destroy (dialog);

    while (gtk_events_pending()) {
		gtk_main_iteration();
    }

    return filename;
}

#endif

const char *fileDialog()
{
#ifdef USE_GTK
    return fileDialogGTK();
#endif
}

}
}
